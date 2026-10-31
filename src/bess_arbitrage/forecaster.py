"""
Electricity price forecasting module.

Trains gradient-boosted regression trees (scikit-learn) with quantile
loss to produce 24-hour-ahead point forecasts and 80% prediction
intervals for day-ahead electricity prices.

Features
--------
- Calendar signals: hour-of-day, day-of-week, month (sin/cos encoded
  for cyclicality)
- Lagged prices: t-24, t-48, t-168  (same hour: yesterday, 2 days ago,
  last week) — all fall on day D-1 or earlier, so no look-ahead
- Rolling mean and std over the previous complete day (midnight-to-midnight
  on day D-1) — available when the bid is placed at noon on day D-1

Three models are trained jointly:
  model_p50  — median forecast (quantile loss, alpha=0.50)
  model_p10  — 10th-percentile lower bound (quantile loss, alpha=0.10)
  model_p90  — 90th-percentile upper bound (quantile loss, alpha=0.90)
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

_MONTH_HOURS = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
_MONTH_END   = list(np.cumsum(_MONTH_HOURS))


def _hour_to_month(t: int) -> int:
    """Map hour index (0-based) to month number (1-12)."""
    for i, end in enumerate(_MONTH_END):
        if t < end:
            return i + 1
    return 12


def _make_features(prices: np.ndarray, t: int) -> list[float]:
    """
    Build feature vector for hour *t* using only information available
    at bid time (noon on day D-1, forecasting hour t on day D).

    Calendar features
    -----------------
    Derived from *t* itself — no price data involved.

    Lagged prices
    -------------
    lag24  = prices[t-24]  (same hour yesterday — day D-1)
    lag48  = prices[t-48]  (same hour two days ago — day D-2)
    lag168 = prices[t-168] (same hour last week)

    All lags are at least 24 hours before *t*, so they are always known
    at bid time regardless of the hour within day D.

    Rolling statistics
    ------------------
    Computed over the 24-hour window prices[day_start-24 : day_start],
    where day_start = t - (t % 24) is the index of midnight at the start
    of day D (index 0 assumed to be 1 Jan 00:00).

    This window covers the previous complete day (D-1, midnight to
    midnight), which is fully observed before noon on day D-1 when bids
    are placed. Prices within day D (indices day_start … t) are NOT used.
    """
    hod = t % 24
    dow = (t // 24) % 7
    mon = _hour_to_month(t) - 1  # 0-indexed for sin/cos encoding

    # Cyclic calendar encoding
    hod_sin = np.sin(2 * np.pi * hod / 24)
    hod_cos = np.cos(2 * np.pi * hod / 24)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    mon_sin = np.sin(2 * np.pi * mon / 12)
    mon_cos = np.cos(2 * np.pi * mon / 12)

    # Lagged prices (same hour: yesterday, two days ago, last week)
    lag24  = float(prices[t - 24])  if t >= 24  else 0.0
    lag48  = float(prices[t - 48])  if t >= 48  else 0.0
    lag168 = float(prices[t - 168]) if t >= 168 else 0.0

    # Rolling statistics over the previous complete day (no same-day prices)
    day_start = t - hod              # index of midnight at start of day D
    w_start   = max(0, day_start - 24)
    window    = prices[w_start:day_start]
    roll_mean = float(window.mean()) if len(window) > 0 else 0.0
    roll_std  = float(window.std())  if len(window) > 1 else 0.0

    return [
        hod_sin, hod_cos,
        dow_sin, dow_cos,
        mon_sin, mon_cos,
        lag24, lag48, lag168,
        roll_mean, roll_std,
    ]


def train_price_forecaster(
    prices: np.ndarray,
    train_end_h: int,
    n_estimators: int = 250,
) -> tuple:
    """
    Train point-forecast and quantile-regression GBM models.

    Parameters
    ----------
    prices      : full hourly price array (EUR/MWh)
    train_end_h : exclusive end hour for training (hours before this are used)
    n_estimators: number of boosting rounds per model

    Returns
    -------
    (model_p50, model_p10, model_p90) — fitted scikit-learn estimators.
    model_p50 uses quantile loss at alpha=0.50 (true median).
    """
    X, y = [], []
    for t in range(168, train_end_h):   # need 168 h history for lag features
        X.append(_make_features(prices, t))
        y.append(float(prices[t]))

    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)

    print(f"[forecaster] Training on {len(y_arr):,} samples (hours 168–{train_end_h})")

    common = dict(
        n_estimators  = n_estimators,
        max_depth     = 5,
        learning_rate = 0.05,
        subsample     = 0.8,
        min_samples_leaf = 10,
        random_state  = 42,
    )
    model_p50 = GradientBoostingRegressor(loss="quantile", alpha=0.50, **common)
    model_p10 = GradientBoostingRegressor(loss="quantile", alpha=0.10, **common)
    model_p90 = GradientBoostingRegressor(loss="quantile", alpha=0.90, **common)

    model_p50.fit(X_arr, y_arr)
    model_p10.fit(X_arr, y_arr)
    model_p90.fit(X_arr, y_arr)

    print("[forecaster] Training complete.")
    return model_p50, model_p10, model_p90


def forecast_prices(
    models: tuple,
    prices: np.ndarray,
    start_h: int,
    end_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 24-hour-ahead forecasts for hours [start_h, end_h).

    Only lag features and previous-day rolling statistics are used;
    no same-day prices enter the feature vector (see _make_features).

    Parameters
    ----------
    models  : (model_p50, model_p10, model_p90) from train_price_forecaster
    prices  : full price array (used for lag/rolling features only)
    start_h : first hour to forecast
    end_h   : exclusive end hour

    Returns
    -------
    (point, lower, upper) — p50, p10, p90 forecasts.
    Quantile order p10 <= p50 <= p90 is enforced per hour by sorting.
    """
    model_p50, model_p10, model_p90 = models
    X = np.array(
        [_make_features(prices, t) for t in range(start_h, end_h)],
        dtype=float,
    )
    p50 = model_p50.predict(X)
    p10 = model_p10.predict(X)
    p90 = model_p90.predict(X)

    # Enforce p10 <= p50 <= p90 per hour (independent models can cross)
    stacked = np.sort(np.stack([p10, p50, p90], axis=1), axis=1)
    return stacked[:, 1], stacked[:, 0], stacked[:, 2]
