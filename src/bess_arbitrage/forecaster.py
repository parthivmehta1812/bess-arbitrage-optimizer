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
  last week)
- Rolling 24-hour mean and standard deviation

Three models are trained jointly:
  model_p50  — median forecast (squared-error loss)
  model_p10  — 10th-percentile lower bound (quantile loss)
  model_p90  — 90th-percentile upper bound (quantile loss)
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
    Build feature vector for hour *t*.

    Uses only information available at forecast time (no look-ahead).
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

    # Rolling 24-hour statistics
    window    = prices[max(0, t - 24):t]
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
    (model_p50, model_p10, model_p90) — fitted scikit-learn estimators
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
    model_p50 = GradientBoostingRegressor(loss="squared_error", **common)
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

    Parameters
    ----------
    models  : (model_p50, model_p10, model_p90) from train_price_forecaster
    prices  : full price array (used for lag features only — no look-ahead)
    start_h : first hour to forecast
    end_h   : exclusive end hour

    Returns
    -------
    (point, lower, upper) — median, 10th-pct, 90th-pct forecasts
    """
    model_p50, model_p10, model_p90 = models
    X = np.array(
        [_make_features(prices, t) for t in range(start_h, end_h)],
        dtype=float,
    )
    return (
        model_p50.predict(X),
        model_p10.predict(X),
        model_p90.predict(X),
    )
