"""
Conformal Quantile Regression (CQR) calibration for electricity price forecasts.

Two variants
------------
- Static CQR  : offsets learned once on a held-out calibration set (e.g. August),
                then applied to all test hours.
- Rolling CQR : offsets recomputed each test day from the preceding 30-day window
                of already-observed actuals vs. forecasts (no look-ahead).

Reference
---------
Angelopoulos & Bates (2021), "A Gentle Introduction to Conformal Prediction
and Distribution-Free Uncertainty Quantification", arXiv:2107.07511.
"""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile_offset(
    y_cal: np.ndarray,
    q_cal: np.ndarray,
    alpha: float,
) -> float:
    """
    One-sided CQR correction for a single quantile level.

    Score  s_i = y_i - q_hat_i
    Offset = empirical quantile of s at level ceil((n+1)*alpha) / n
             (finite-sample coverage guarantee).
    Apply  q_new_i = q_hat_i + offset.

    Parameters
    ----------
    y_cal : actual values on the calibration set, shape (n,)
    q_cal : predicted alpha-quantile on the calibration set, shape (n,)
    alpha : target quantile level, e.g. 0.10 or 0.90

    Returns
    -------
    float scalar offset to add to test-set quantile forecasts
    """
    y_cal = np.asarray(y_cal, dtype=float)
    q_cal = np.asarray(q_cal, dtype=float)
    n = len(y_cal)
    scores = y_cal - q_cal
    level = min(math.ceil((n + 1) * alpha) / n, 1.0)
    return float(np.quantile(scores, level))


def rolling_conformal(
    y_cal: np.ndarray,
    q10_cal: np.ndarray,
    q90_cal: np.ndarray,
    y_test: np.ndarray,
    q10_test: np.ndarray,
    q50_test: np.ndarray,
    q90_test: np.ndarray,
    window_days: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive CQR: for each test day, refit p10 and p90 offsets from the
    preceding `window_days` days of actuals vs. forecasts.

    Uses calibration data to seed the window for the first test days.
    No look-ahead: day d uses only data from days strictly before d.

    Parameters
    ----------
    y_cal, q10_cal, q90_cal  : calibration actuals + quantile forecasts (hourly)
    y_test, q10_test, q50_test, q90_test : test actuals + quantile forecasts
    window_days : rolling window length in days (default 30)

    Returns
    -------
    (q10_out, q50_out, q90_out) — calibrated arrays for the test period,
    with p10 <= p50 <= p90 enforced per hour.
    """
    y_cal    = np.asarray(y_cal,    dtype=float)
    q10_cal  = np.asarray(q10_cal,  dtype=float)
    q90_cal  = np.asarray(q90_cal,  dtype=float)
    y_test   = np.asarray(y_test,   dtype=float)
    q10_test = np.asarray(q10_test, dtype=float)
    q50_test = np.asarray(q50_test, dtype=float)
    q90_test = np.asarray(q90_test, dtype=float)

    # Concatenate calibration then test for rolling lookback
    y_all   = np.concatenate([y_cal,   y_test])
    q10_all = np.concatenate([q10_cal, q10_test])
    q90_all = np.concatenate([q90_cal, q90_test])

    n_cal_days  = len(y_cal)  // 24
    n_test_days = len(y_test) // 24

    q10_out = q10_test.copy()
    q90_out = q90_test.copy()

    for d in range(n_test_days):
        # global_day is the day index of test day d in the concatenated array
        global_day    = n_cal_days + d
        win_start_day = max(0, global_day - window_days)
        # window is [win_start_day, global_day) — strictly before day d
        h_start = win_start_day * 24
        h_end   = global_day   * 24

        y_w   = y_all[h_start:h_end]
        q10_w = q10_all[h_start:h_end]
        q90_w = q90_all[h_start:h_end]

        if len(y_w) == 0:
            continue

        off10 = conformal_quantile_offset(y_w, q10_w, 0.10)
        off90 = conformal_quantile_offset(y_w, q90_w, 0.90)

        h0 = d * 24
        h1 = h0 + 24
        q10_out[h0:h1] = q10_test[h0:h1] + off10
        q90_out[h0:h1] = q90_test[h0:h1] + off90

    # Enforce p10 <= p50 <= p90 per hour
    stacked = np.sort(np.stack([q10_out, q50_test, q90_out], axis=1), axis=1)
    return stacked[:, 0], q50_test.copy(), stacked[:, 2]
