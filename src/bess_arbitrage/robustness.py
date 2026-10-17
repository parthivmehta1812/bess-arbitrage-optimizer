"""
Robustness layer for forecast-driven BESS dispatch.

Components
----------
1. Plausibility checks on the price history used as forecast INPUT.
   Flags NaN, out-of-bounds, spikes (modified z-score), and stuck values.
   Flagged values are set to NaN; counts are returned per type.

2. Missing-data imputation:
   - Short gaps (<= 3 h): linear interpolation (uses adjacent valid values).
   - Longer gaps: same hour one week earlier (t-168) if valid, else t-24.
   Long-gap lookbacks are strictly backward (no future data).

3. Day-level forecast fallback:
   For each 24-hour test day, decide between GBM p50 and persistence.
   Fall back to persistence when:
     a) low confidence: day's mean (p90-p10) > k × rolling-30-day median, or
     b) high imputation: > 25 % of that day's lag inputs (t-24, t-48, t-168)
        were imputed.
   k is selected from {1.25, 1.5, 2.0} on January–February ONLY.

References
----------
Iglewicz & Hoaglin (1993) "How to Detect and Handle Outliers" for the
modified z-score with MAD.
"""

from __future__ import annotations

import numpy as np

from .forecaster import train_price_forecaster, forecast_prices

# ── Constants ─────────────────────────────────────────────────────────────────
PRICE_LO       = -500.0   # Day-ahead market floor  (EUR/MWh, from 2021)
PRICE_HI       = 4000.0   # Day-ahead market ceiling (EUR/MWh, from 2021)
STUCK_MIN_RUN  = 6        # Minimum consecutive identical hours to flag as stuck
SPIKE_Z_THRESH = 3.5      # Modified z-score threshold for spike detection
SPIKE_WINDOW   = 168      # Rolling window length (7 days = 168 h)
SHORT_GAP_MAX  = 3        # Maximum gap length for linear interpolation

# K-calibration boundaries (Jan–Feb 2023 in hourly 2023 data)
_K_CAL_TRAIN_END = 744    # Jan end / Feb start
_K_CAL_END       = 1416   # Feb end / Mar start (= first test month start)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Plausibility checks
# ─────────────────────────────────────────────────────────────────────────────

def check_and_flag(prices: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Detect and flag problematic values in a price array.

    Checks applied (on the original, unflagged values):
      1. Pre-existing NaN
      2. Out-of-market-bounds: value < PRICE_LO or > PRICE_HI
      3. Spikes: |modified z-score| > SPIKE_Z_THRESH on a rolling
         SPIKE_WINDOW-hour backward window  (min 4 non-NaN points)
      4. Stuck values: identical non-NaN value for >= STUCK_MIN_RUN
         consecutive hours

    Flagged values are set to NaN in the returned array.

    Parameters
    ----------
    prices : raw input price array, may already contain NaN

    Returns
    -------
    (flagged_prices, report)
    report : dict with keys n_nan, n_bounds, n_spike, n_stuck
    """
    prices = np.asarray(prices, dtype=float)
    n      = len(prices)
    result = prices.copy()

    # 1. Count pre-existing NaN (do not double-count below)
    nan_orig = np.isnan(result)
    n_nan    = int(nan_orig.sum())

    # 2. Bounds check (only on currently valid values)
    valid = ~nan_orig
    bounds = np.zeros(n, dtype=bool)
    bounds[valid] = (result[valid] < PRICE_LO) | (result[valid] > PRICE_HI)
    n_bounds = int(bounds.sum())
    result[bounds] = np.nan

    # 3. Spikes: rolling modified z-score (causal window, backward only)
    spike = np.zeros(n, dtype=bool)
    for t in range(n):
        if np.isnan(result[t]):
            continue
        start  = max(0, t - SPIKE_WINDOW + 1)
        window = result[start : t + 1]
        window = window[~np.isnan(window)]
        if len(window) < 4:          # too few points for robust estimate
            continue
        med = float(np.median(window))
        mad = float(np.median(np.abs(window - med)))
        if mad < 1e-8:               # all window values identical → not a spike
            continue
        z = 0.6745 * (result[t] - med) / mad
        if abs(z) > SPIKE_Z_THRESH:
            spike[t] = True
    n_spike = int(spike.sum())
    result[spike] = np.nan

    # 4. Stuck values: runs of >= STUCK_MIN_RUN identical non-NaN values
    stuck = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if np.isnan(result[i]):
            i += 1
            continue
        j = i + 1
        while j < n and result[j] == result[i]:
            j += 1
        if j - i >= STUCK_MIN_RUN:
            stuck[i:j] = True
        i = j
    n_stuck = int(stuck.sum())
    result[stuck] = np.nan

    report = {
        "n_nan":    n_nan,
        "n_bounds": n_bounds,
        "n_spike":  n_spike,
        "n_stuck":  n_stuck,
    }
    return result, report


# ─────────────────────────────────────────────────────────────────────────────
# 2. Imputation
# ─────────────────────────────────────────────────────────────────────────────

def impute_gaps(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill NaN gaps.

    Strategy
    --------
    - Gaps <= SHORT_GAP_MAX (3 h): linear interpolation between the adjacent
      valid values (uses both left and right neighbour in batch).
      If only a left (right) neighbour exists, forward (backward) fill.
    - Longer gaps: prices[t-168] if valid, else prices[t-24].
      Both lookbacks are strictly backward — they never use data from
      positions > t.
    - Values that cannot be filled remain NaN.

    Parameters
    ----------
    prices : array possibly containing NaN

    Returns
    -------
    (imputed_prices, imputed_mask)
    imputed_mask[t] = True if prices[t] was NaN and was subsequently filled.
    """
    result       = prices.copy()
    n            = len(result)
    imputed_mask = np.zeros(n, dtype=bool)

    # Short-gap linear interpolation
    i = 0
    while i < n:
        if not np.isnan(result[i]):
            i += 1
            continue
        # Gap starts at i; find end j (exclusive)
        j = i
        while j < n and np.isnan(result[j]):
            j += 1
        gap_len = j - i
        if gap_len <= SHORT_GAP_MAX:
            left  = result[i - 1] if i > 0 and not np.isnan(result[i - 1]) else None
            right = result[j]     if j < n and not np.isnan(result[j])     else None
            if left is not None and right is not None:
                for k in range(gap_len):
                    result[i + k]       = left + (right - left) * (k + 1) / (gap_len + 1)
                    imputed_mask[i + k] = True
            elif left is not None:
                result[i:j]       = left
                imputed_mask[i:j] = True
            elif right is not None:
                result[i:j]       = right
                imputed_mask[i:j] = True
            # Both None → gap remains; will be handled by long-gap fallback
        i = j

    # Long-gap fallback: t-168 (same hour last week), then t-24 (yesterday)
    for t in range(n):
        if not np.isnan(result[t]):
            continue
        if t >= 168 and not np.isnan(result[t - 168]):
            result[t]       = result[t - 168]
            imputed_mask[t] = True
        elif t >= 24 and not np.isnan(result[t - 24]):
            result[t]       = result[t - 24]
            imputed_mask[t] = True
        # If still NaN: no valid backward fallback; remains NaN (very rare)

    return result, imputed_mask


def clean_prices(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Combined plausibility-check + imputation pipeline.

    Parameters
    ----------
    prices : raw input price array

    Returns
    -------
    (clean_prices, imputed_mask, report)
    clean_prices  : flagged values replaced, gaps imputed; any remaining NaN
                    replaced with 0.0 as last resort (extremely rare)
    imputed_mask  : bool array, True where values were imputed
    report        : flag counts (n_nan, n_bounds, n_spike, n_stuck)
    """
    flagged, report   = check_and_flag(prices)
    imputed, mask     = impute_gaps(flagged)
    # Last-resort NaN replacement (keeps forecaster from receiving NaN features)
    still_nan         = np.isnan(imputed)
    imputed[still_nan] = 0.0
    return imputed, mask, report


def naive_ffill(prices: np.ndarray) -> np.ndarray:
    """
    Forward-fill NaN values without any plausibility checks.
    Used by strategy A (baseline, no robustness layer).
    If the array starts with NaN, those remain NaN until a valid value appears.
    """
    result = prices.copy()
    last   = np.nan
    for t in range(len(result)):
        if np.isnan(result[t]):
            result[t] = last      # may still be NaN if no prior valid value
        else:
            last = result[t]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Confidence / fallback utilities
# ─────────────────────────────────────────────────────────────────────────────

def daily_interval_widths(p10: np.ndarray, p90: np.ndarray) -> np.ndarray:
    """
    Mean (p90 − p10) per 24-hour day.

    Parameters
    ----------
    p10, p90 : quantile forecasts, shape (T,)  where T is a multiple of 24

    Returns
    -------
    np.ndarray of shape (T // 24,)
    """
    n_days = len(p10) // 24
    return np.array([
        float(np.mean(p90[d * 24:(d + 1) * 24] - p10[d * 24:(d + 1) * 24]))
        for d in range(n_days)
    ])


def choose_k(
    prices: np.ndarray,
    k_candidates: tuple[float, ...] = (1.25, 1.5, 2.0),
    target_rate: float = 0.15,
) -> float:
    """
    Select the fallback threshold k using ONLY January–February data.

    Method
    ------
    1. Train a fast GBM (n_estimators=100) on January (hours 168–744).
    2. Forecast February (hours 744–1416) → p10, p90 arrays.
    3. Compute 28 daily interval widths for February.
    4. For each k in k_candidates, compute fallback_rate = fraction of February
       days where width > k × median(February widths).
    5. Return the k whose fallback_rate is closest to target_rate.
    The test period (hours 1416–8760) is never touched.

    Parameters
    ----------
    prices       : CLEAN full-year price array, shape (8760,)
    k_candidates : candidate threshold values
    target_rate  : desired fraction of calibration days that trigger fallback

    Returns
    -------
    Chosen k (one of k_candidates)
    """
    models = train_price_forecaster(prices, train_end_h=_K_CAL_TRAIN_END, n_estimators=100)
    _, p10_feb, p90_feb = forecast_prices(models, prices, _K_CAL_TRAIN_END, _K_CAL_END)
    widths   = daily_interval_widths(p10_feb, p90_feb)   # 28 Feb days

    if len(widths) == 0:
        return float(k_candidates[len(k_candidates) // 2])

    median_w = float(np.median(widths))
    if median_w < 1e-6:
        return float(k_candidates[len(k_candidates) // 2])

    best_k   = float(k_candidates[len(k_candidates) // 2])
    best_err = float("inf")
    for k in k_candidates:
        rate = float(np.mean(widths > k * median_w))
        err  = abs(rate - target_rate)
        if err < best_err:
            best_err = err
            best_k   = float(k)
    return best_k


def select_forecast_per_day(
    p50: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
    persistence: np.ndarray,
    imputed_mask: np.ndarray,
    hist_widths: np.ndarray,
    k: float,
    h_offset: int = 0,
    imputed_thresh: float = 0.25,
) -> tuple[np.ndarray, list[str]]:
    """
    Select the dispatch forecast signal day by day: GBM p50 or persistence.

    For each 24-hour day d:
      day_width = mean(p90[d*24:(d+1)*24] − p10[d*24:(d+1)*24])
      If len(hist_widths) >= 7:
          low_conf = day_width > k × median(last 30 hist_widths)
      Else: low_conf = False (insufficient history)
      For each hour h in the day, check lag inputs:
          any_lag_imp[h] = imputed_mask[h_offset+h-24] OR [h_offset+h-48]
                           OR [h_offset+h-168]  (only if index >= 0)
      high_imp = mean(any_lag_imp) > imputed_thresh
      branch = "persistence" if (low_conf or high_imp) else "gbm"

    Parameters
    ----------
    p50, p10, p90   : GBM forecasts for one test month, shape (T,)
    persistence     : cleaned persistence forecast for same period, shape (T,)
    imputed_mask    : bool array covering the FULL price history, shape (N,)
                      N >= h_offset + T + max_lag (168).
                      imputed_mask[t] = True if the value at absolute hour t
                      was imputed during cleaning.
    hist_widths     : daily interval widths from BEFORE this test period
    k               : confidence threshold (chosen on Jan–Feb only)
    h_offset        : absolute hour of the first element of p50/p10/p90
    imputed_thresh  : fraction of imputed lag inputs that triggers fallback

    Returns
    -------
    (selected_forecast, branches)
    selected_forecast : shape (T,)
    branches          : list of "gbm" or "persistence", one entry per day
    """
    T      = len(p50)
    n_days = T // 24

    selected = np.empty(T, dtype=float)
    branches: list[str] = []

    hist = list(hist_widths) if len(hist_widths) > 0 else []

    for d in range(n_days):
        h0 = d * 24
        h1 = h0 + 24

        # Confidence check
        day_width = float(np.mean(p90[h0:h1] - p10[h0:h1]))
        if len(hist) >= 7:
            recent   = hist[-30:] if len(hist) >= 30 else hist
            med_w    = float(np.median(recent))
            low_conf = (med_w > 0) and (day_width > k * med_w)
        else:
            low_conf = False

        # Imputation check: fraction of test hours whose lag inputs were imputed
        N_mask  = len(imputed_mask)
        day_imp = np.zeros(24, dtype=bool)
        for i_h in range(24):
            h_abs = h_offset + h0 + i_h
            for lag in (24, 48, 168):
                idx = h_abs - lag
                if 0 <= idx < N_mask and imputed_mask[idx]:
                    day_imp[i_h] = True
                    break
        high_imp = float(day_imp.mean()) > imputed_thresh

        branch = "persistence" if (low_conf or high_imp) else "gbm"
        branches.append(branch)
        selected[h0:h1] = persistence[h0:h1] if branch == "persistence" else p50[h0:h1]

    return selected, branches
