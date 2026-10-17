"""
Walk-forward (rolling-origin) backtest for BESS arbitrage with GBM price forecasts.

Design
------
- Expanding window, monthly re-training:
    for each test month m = March ... December (2023 DE prices):
        - train GBM forecaster on hours [168, start_of(m))
        - produce 24h-ahead p10/p50/p90 forecasts for month m
        - naive persistence baseline: price[t-24]
- Economic evaluation (per month, three scenarios):
    - Perfect foresight : MILP on actual prices  → revenue on actual prices
    - GBM               : MILP on p50 forecast   → revenue on actual prices
    - Persistence       : MILP on pers. forecast → revenue on actual prices
- Forecast-quality metrics (GBM vs actuals):
    MAE, RMSE, p90 coverage, 80% interval coverage, CRPS
- Diebold–Mariano test (GBM p50 vs persistence, test months pooled):
    daily loss differentials, Newey–West (lag=1), HLN small-sample correction

The MILP is built directly from a price array (no loader padding), so monthly
optimizations are clean and the terminal SoC constraint is at the correct month end.

BESS parameters (fixed): 8 MWh / 2 MW / η=0.92 / initial SoC 0.5
  — same as the main arbitrage example.

Reference
---------
Diebold & Mariano (1995), "Comparing Predictive Accuracy", JBES.
Harvey, Leybourne & Newbold (1997), "Testing the equality of prediction mean
  squared errors", IJF.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np
import pandas as pd

from .forecaster import train_price_forecaster, forecast_prices

# ── 2023 month boundaries (non-leap year, same as forecaster._MONTH_HOURS) ──
_MONTH_HOURS = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
_MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
_MONTH_STARTS: list[int] = [0]
for _h in _MONTH_HOURS:
    _MONTH_STARTS.append(_MONTH_STARTS[-1] + _h)
# _MONTH_STARTS[0..12]:
#   [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]


def _pinball(alpha: float, y: np.ndarray, q: np.ndarray) -> float:
    return float(np.where(y >= q, alpha * (y - q), (1 - alpha) * (q - y)).mean())


def _milp_schedule(
    prices: np.ndarray,
    battery_capacity_mwh: float,
    max_power_mw: float,
    efficiency: float,
    initial_soc: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the arbitrage MILP on an arbitrary-length price array.

    Builds the Pyomo model directly from the supplied price vector — no
    loader, no padding to 8760 hours.  The terminal SoC constraint is
    placed at the last hour of the supplied array.

    Parameters
    ----------
    prices               : hourly price array, any length T ≥ 2
    battery_capacity_mwh : fixed battery capacity [MWh]
    max_power_mw         : max charge/discharge rate [MW]
    efficiency           : round-trip efficiency (0–1)
    initial_soc          : initial SoC as fraction of capacity

    Returns
    -------
    (charge_arr, discharge_arr) — numpy arrays of shape (T,) in MW
    """
    try:
        from pyomo.environ import (
            Binary, ConcreteModel, Constraint, NonNegativeReals,
            Objective, Set, SolverFactory, Var, maximize, value,
        )
    except ImportError:
        raise ImportError(
            "Pyomo is not installed. Install with:  pip install pyomo highspy"
        )

    prices = np.asarray(prices, dtype=float)
    T = len(prices)
    sqrt_eff = math.sqrt(efficiency)
    cap   = battery_capacity_mwh
    power = max_power_mw

    m = ConcreteModel()
    m.T = Set(initialize=range(T))

    m.charge      = Var(m.T, domain=NonNegativeReals, bounds=(0, power))
    m.discharge   = Var(m.T, domain=NonNegativeReals, bounds=(0, power))
    m.soc         = Var(m.T, domain=NonNegativeReals, bounds=(0, cap))
    m.is_charging = Var(m.T, domain=Binary)

    m.obj = Objective(
        rule=lambda m: sum(
            prices[t] * (m.discharge[t] * sqrt_eff - m.charge[t] / sqrt_eff)
            for t in m.T
        ),
        sense=maximize,
    )

    # Initial SoC
    m.soc_init = Constraint(rule=lambda m: m.soc[0] == initial_soc * cap)

    # SoC dynamics
    def _soc_dyn(m, t):
        if t == 0:
            return Constraint.Skip
        return (
            m.soc[t]
            == m.soc[t - 1]
            + m.charge[t - 1] * sqrt_eff
            - m.discharge[t - 1] / sqrt_eff
        )
    m.soc_dynamics = Constraint(m.T, rule=_soc_dyn)

    # Terminal SoC >= 50 % (prevents end-of-period drain)
    m.soc_terminal = Constraint(rule=lambda m: m.soc[T - 1] >= 0.5 * cap)

    # Mutex: no simultaneous charge + discharge (big-M)
    big_m = power * 2
    m.mutex_charge = Constraint(
        m.T, rule=lambda m, t: m.charge[t] <= big_m * m.is_charging[t]
    )
    m.mutex_discharge = Constraint(
        m.T, rule=lambda m, t: m.discharge[t] <= big_m * (1 - m.is_charging[t])
    )

    solver = SolverFactory("highs")
    if not solver.available():
        solver = SolverFactory("cbc")
    solver.solve(m, tee=False, options={"time_limit": 120})

    charge_arr    = np.asarray([value(m.charge[t])    for t in range(T)], dtype=float)
    discharge_arr = np.asarray([value(m.discharge[t]) for t in range(T)], dtype=float)
    charge_arr    = np.clip(charge_arr,    0, power)
    discharge_arr = np.clip(discharge_arr, 0, power)
    return charge_arr, discharge_arr


def _actual_revenue(
    actual_prices: np.ndarray,
    charge_arr: np.ndarray,
    discharge_arr: np.ndarray,
    efficiency: float,
) -> float:
    """Revenue on *actual_prices* using the supplied charge/discharge schedule."""
    sqrt_eff = math.sqrt(efficiency)
    return float(
        np.sum(actual_prices * (discharge_arr * sqrt_eff - charge_arr / sqrt_eff))
    )


def walk_forward_backtest(
    prices: np.ndarray,
    first_test_month: int = 3,
    battery_capacity_mwh: float = 8.0,
    max_power_mw: float = 2.0,
    efficiency: float = 0.92,
    initial_soc: float = 0.5,
    n_estimators: int = 250,
) -> pd.DataFrame:
    """
    Walk-forward expanding-window backtest over monthly test periods.

    For each test month m in [first_test_month, 12]:
      1. Train GBM forecaster on hours [168, _MONTH_STARTS[m-1]).
         Training data ends strictly before the first test hour.
      2. Forecast month m — returns (p50, p10, p90) arrays.
      3. Persistence baseline: prices[t-24] for each test hour t.
      4. Run three MILP optimizations, all evaluated on actual prices:
           - Perfect foresight : schedule from actual prices
           - GBM               : schedule from p50 forecast
           - Persistence       : schedule from persistence forecast
      5. Compute forecast-quality metrics (GBM vs actuals):
           MAE, RMSE, p90 coverage, 80% interval coverage, CRPS.

    Parameters
    ----------
    prices             : full-year hourly price array (8760,)
    first_test_month   : first 1-indexed month to test (default 3 = March)
    battery_capacity_mwh, max_power_mw, efficiency, initial_soc :
                         BESS parameters (same defaults as main examples)
    n_estimators       : GBM boosting rounds (default 250)

    Returns
    -------
    pandas DataFrame — one row per test month + one "Overall" row.

    Columns
    -------
    month, n_hours,
    mae, rmse,
    p90_cov_pct, int80_cov_pct, crps,
    mae_pers, rmse_pers,
    revenue_perfect_eur, revenue_gbm_eur, revenue_pers_eur,
    capture_gbm_pct, capture_pers_pct
    """
    prices = np.asarray(prices, dtype=float)

    bess_kw = dict(
        battery_capacity_mwh=battery_capacity_mwh,
        max_power_mw=max_power_mw,
        efficiency=efficiency,
        initial_soc=initial_soc,
    )

    rows: list[dict] = []

    for m_idx in range(first_test_month - 1, 12):   # 0-indexed
        month_name = _MONTH_NAMES[m_idx]
        h_start    = _MONTH_STARTS[m_idx]
        h_end      = _MONTH_STARTS[m_idx + 1]
        actual     = prices[h_start:h_end]

        # ── 1. Train on hours [168, h_start) — strictly before test month ──
        print(f"\n[{month_name}] Training forecaster on hours 168–{h_start} …", flush=True)
        models = train_price_forecaster(
            prices, train_end_h=h_start, n_estimators=n_estimators
        )

        # ── 2. GBM forecast for this month ───────────────────────────────
        print(f"[{month_name}] Forecasting hours {h_start}–{h_end} …", flush=True)
        p50, p10, p90 = forecast_prices(models, prices, h_start, h_end)

        # ── 3. Persistence: price[t-24] for each test hour ───────────────
        pers = prices[h_start - 24 : h_end - 24]   # always valid for m >= 3

        # ── 4. Economic evaluation ────────────────────────────────────────
        print(f"[{month_name}] MILP — perfect foresight …", flush=True)
        c_pf, d_pf = _milp_schedule(actual, **bess_kw)
        rev_perfect = _actual_revenue(actual, c_pf, d_pf, efficiency)

        print(f"[{month_name}] MILP — GBM forecast …", flush=True)
        c_gbm, d_gbm = _milp_schedule(p50, **bess_kw)
        rev_gbm = _actual_revenue(actual, c_gbm, d_gbm, efficiency)

        print(f"[{month_name}] MILP — persistence …", flush=True)
        c_pers, d_pers = _milp_schedule(pers, **bess_kw)
        rev_pers = _actual_revenue(actual, c_pers, d_pers, efficiency)

        # ── 5. Forecast-quality metrics ───────────────────────────────────
        mae      = float(np.mean(np.abs(actual - p50)))
        rmse     = float(np.sqrt(np.mean((actual - p50) ** 2)))
        mae_pers  = float(np.mean(np.abs(actual - pers)))
        rmse_pers = float(np.sqrt(np.mean((actual - pers) ** 2)))

        p90_cov   = float(np.mean(actual <= p90)) * 100.0
        int80_cov = float(np.mean((actual >= p10) & (actual <= p90))) * 100.0

        pb10 = _pinball(0.10, actual, p10)
        pb50 = _pinball(0.50, actual, p50)
        pb90 = _pinball(0.90, actual, p90)
        crps = 2.0 * float(np.mean([pb10, pb50, pb90]))

        capture_gbm  = (rev_gbm  / rev_perfect * 100.0) if rev_perfect > 0 else float("nan")
        capture_pers = (rev_pers / rev_perfect * 100.0) if rev_perfect > 0 else float("nan")

        rows.append({
            "month":               month_name,
            "n_hours":             h_end - h_start,
            "mae":                 mae,
            "rmse":                rmse,
            "p90_cov_pct":         p90_cov,
            "int80_cov_pct":       int80_cov,
            "crps":                crps,
            "mae_pers":            mae_pers,
            "rmse_pers":           rmse_pers,
            "revenue_perfect_eur": rev_perfect,
            "revenue_gbm_eur":     rev_gbm,
            "revenue_pers_eur":    rev_pers,
            "capture_gbm_pct":     capture_gbm,
            "capture_pers_pct":    capture_pers,
        })
        print(
            f"[{month_name}] done.  "
            f"MAE {mae:.1f} €/MWh  |  "
            f"capture GBM {capture_gbm:.1f}%  pers {capture_pers:.1f}%",
            flush=True,
        )

    # ── Overall row (pool all test hours) ────────────────────────────────────
    df = pd.DataFrame(rows)
    w  = df["n_hours"].values.astype(float)

    total_perfect = float(df["revenue_perfect_eur"].sum())
    total_gbm     = float(df["revenue_gbm_eur"].sum())
    total_pers    = float(df["revenue_pers_eur"].sum())

    overall = {
        "month":               "Overall",
        "n_hours":             int(w.sum()),
        "mae":                 float(np.average(df["mae"].values,  weights=w)),
        "rmse":                float(np.sqrt(np.average(df["rmse"].values ** 2, weights=w))),
        "p90_cov_pct":         float(np.average(df["p90_cov_pct"].values, weights=w)),
        "int80_cov_pct":       float(np.average(df["int80_cov_pct"].values, weights=w)),
        "crps":                float(np.average(df["crps"].values, weights=w)),
        "mae_pers":            float(np.average(df["mae_pers"].values, weights=w)),
        "rmse_pers":           float(np.sqrt(np.average(df["rmse_pers"].values ** 2, weights=w))),
        "revenue_perfect_eur": total_perfect,
        "revenue_gbm_eur":     total_gbm,
        "revenue_pers_eur":    total_pers,
        "capture_gbm_pct":     (total_gbm  / total_perfect * 100.0) if total_perfect > 0 else float("nan"),
        "capture_pers_pct":    (total_pers / total_perfect * 100.0) if total_perfect > 0 else float("nan"),
    }

    return pd.concat([df, pd.DataFrame([overall])], ignore_index=True)


def diebold_mariano(
    e1: np.ndarray,
    e2: np.ndarray,
    loss: Literal["abs", "sq"] = "abs",
    lag: int | None = 1,
) -> dict:
    """
    Diebold–Mariano test on daily mean loss differentials.

    Tests H1: model 2 has lower expected loss than model 1 (one-sided).
    i.e. H1: E[L(e1)] > E[L(e2)], equivalently E[d] > 0 where
    d_day = mean over 24 hours of [L(e1) − L(e2)].

    Forecast errors within a day are correlated for 24h-ahead predictions,
    so we aggregate to daily mean loss differentials before computing the
    test statistic.  The Newey–West long-run variance with `lag` lags on
    the daily series accounts for residual serial correlation.

    The Harvey–Leybourne–Newbold (1997) small-sample correction multiplies
    the DM statistic by sqrt((n+1−2h+h(h−1)/n)/n) where n = number of days
    and h = 1 (one-step-ahead on the daily scale).  The HLN statistic is
    compared against a t(n−1) distribution.

    Parameters
    ----------
    e1, e2  : 1-D arrays of hourly forecast errors, length n_days × 24.
              Errors are actual − forecast (sign does not affect loss values).
              e1 should be the *less accurate* model (e.g. persistence),
              e2 the model under test (e.g. GBM).
    loss    : "abs" for mean absolute error loss, "sq" for squared-error loss
    lag     : Newey–West lag on the daily series (default 1; None → no NW correction)

    Returns
    -------
    dict with keys:
        dm_stat      : DM test statistic (positive → model 2 better than model 1)
        p_normal     : one-sided p-value under standard normal approximation
        hln_stat     : Harvey–Leybourne–Newbold corrected statistic
        p_hln        : one-sided p-value under t(n_days−1)
        n_days       : number of daily differentials used
        loss_mean_e1 : mean hourly loss for model 1
        loss_mean_e2 : mean hourly loss for model 2
    """
    try:
        from scipy import stats as _stats
    except ImportError:
        raise ImportError(
            "scipy is required for the Diebold–Mariano test. "
            "Install with:  pip install scipy"
        )

    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)

    if len(e1) != len(e2):
        raise ValueError(
            f"e1 and e2 must have equal length, got {len(e1)} vs {len(e2)}"
        )

    n_hours = len(e1)
    n_days  = n_hours // 24
    if n_days == 0:
        raise ValueError("Need at least 24 hourly errors (1 complete day).")

    # Truncate to full days
    e1 = e1[: n_days * 24]
    e2 = e2[: n_days * 24]

    if loss == "abs":
        L1 = np.abs(e1)
        L2 = np.abs(e2)
    elif loss == "sq":
        L1 = e1 ** 2
        L2 = e2 ** 2
    else:
        raise ValueError(f"loss must be 'abs' or 'sq', got {loss!r}")

    loss_mean_e1 = float(L1.mean())
    loss_mean_e2 = float(L2.mean())

    # Daily loss differentials d_day = mean over 24h of (L1 − L2)
    d = (L1 - L2).reshape(n_days, 24).mean(axis=1)   # shape (n_days,)
    d_mean = float(d.mean())

    n = n_days
    lag_nw = max(0, int(lag)) if lag is not None else 0

    # Newey–West long-run variance (Bartlett kernel)
    d_dev   = d - d_mean
    gamma_0 = float(np.mean(d_dev ** 2))
    lrv     = gamma_0
    for k in range(1, lag_nw + 1):
        w_k      = 1.0 - k / (lag_nw + 1)
        gamma_k  = float(np.mean(d_dev[k:] * d_dev[:n - k]))
        lrv     += 2.0 * w_k * gamma_k
    lrv = max(lrv, 0.0)    # numerical safety

    if lrv == 0.0:
        warnings.warn(
            "DM long-run variance is zero — forecasts may be identical "
            "or differences have zero variance.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "dm_stat":      float("nan"),
            "p_normal":     float("nan"),
            "hln_stat":     float("nan"),
            "p_hln":        float("nan"),
            "n_days":       n_days,
            "loss_mean_e1": loss_mean_e1,
            "loss_mean_e2": loss_mean_e2,
        }

    dm_stat  = d_mean / math.sqrt(lrv / n)
    p_normal = float(_stats.norm.sf(dm_stat))   # P(Z > dm_stat), one-sided

    # HLN correction: h=1 on daily scale → factor = sqrt((n-1)/n)
    h          = 1
    hln_factor = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    hln_stat   = dm_stat * hln_factor
    p_hln      = float(_stats.t.sf(hln_stat, df=n - 1))

    return {
        "dm_stat":      dm_stat,
        "p_normal":     p_normal,
        "hln_stat":     hln_stat,
        "p_hln":        p_hln,
        "n_days":       n_days,
        "loss_mean_e1": loss_mean_e1,
        "loss_mean_e2": loss_mean_e2,
    }
