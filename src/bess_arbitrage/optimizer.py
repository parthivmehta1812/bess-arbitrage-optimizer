"""
BESS Arbitrage Optimizer
========================
Mixed-Integer Linear Program (MILP) using Pyomo + HiGHS / CBC.

Objective
---------
Maximise annual revenue by charging a battery when grid prices are low
and discharging when prices are high.

Decision variables (per hour *t*)
----------------------------------
charge[t]       MW drawn from grid into battery
discharge[t]    MW exported from battery to grid
soc[t]          MWh stored (State of Charge)
is_charging[t]  binary mutex  (1 = charging, 0 = discharging)

Optional sizing mode (``optimize_sizing=True``)
------------------------------------------------
battery_cap     continuous variable (MWh) solved by the optimizer
power           continuous variable (MW) solved by the optimizer
E:P ratio kept within [ep_ratio_min, ep_ratio_max] hours to enforce
a physically realistic battery duration (default 2–4 h).
"""

from __future__ import annotations

import math

import numpy as np

try:
    from pyomo.environ import (
        Binary,
        ConcreteModel,
        Constraint,
        NonNegativeReals,
        Objective,
        Set,
        SolverFactory,
        Var,
        maximize,
        value,
    )
    _PYOMO_OK = True
except ImportError:
    _PYOMO_OK = False

from .loader import load_prices


def make_industrial_load(
    peak_mw: float = 2.2,
    base_mw: float = 0.4,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic 8,760-hour industrial electricity demand profile.

    Mimics a typical manufacturing facility with:
    - Day/night and weekday/weekend patterns
    - Afternoon demand peaks (shift changeover)
    - Gaussian noise for realistic variability

    Parameters
    ----------
    peak_mw : float
        Maximum demand during production hours [MW].
    base_mw : float
        Baseline overnight demand (HVAC, lighting, standby) [MW].
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (8760,), units MW.
    """
    rng = np.random.default_rng(seed)
    # Hourly shape factor for a weekday (0 = midnight, 23 = 11pm)
    weekday_shape = np.array([
        0.22, 0.20, 0.19, 0.19, 0.20, 0.30,   # 0–5   night / pre-dawn
        0.55, 0.82, 0.95, 0.98, 1.00, 0.97,   # 6–11  morning ramp + full production
        0.90, 0.93, 0.96, 1.00, 0.98, 0.92,   # 12–17 afternoon peak
        0.75, 0.60, 0.45, 0.35, 0.28, 0.24,   # 18–23 wind-down
    ])
    weekend_shape = weekday_shape * 0.40       # skeleton crew on weekends

    profile = np.empty(8760)
    for h in range(8760):
        dow = (h // 24) % 7
        hod = h % 24
        shape = weekday_shape[hod] if dow < 5 else weekend_shape[hod]
        noise = rng.normal(0, 0.03)
        raw = base_mw + (peak_mw - base_mw) * shape + noise
        profile[h] = np.clip(raw, base_mw * 0.8, peak_mw * 1.05)

    return profile


def optimize_bess_arbitrage(
    price_bytes: bytes | None = None,
    battery_capacity_mwh: float = 8.0,
    max_power_mw: float = 2.0,
    efficiency: float = 0.92,
    max_grid_mw: float = 10.0,
    optimize_sizing: bool = False,
    ep_ratio_min: float = 2.0,
    ep_ratio_max: float = 4.0,
    rfnbo_compliant: bool = False,
    initial_soc: float = 0.5,
) -> dict:
    """
    Run the BESS arbitrage MILP and return results.

    Parameters
    ----------
    price_bytes : bytes
        Raw bytes of a CSV or XLSX file with hourly electricity prices
        (EUR/MWh).  Pass ``None`` to use the bundled 2024 DE sample.
    battery_capacity_mwh : float
        Fixed battery size in MWh.  Ignored when *optimize_sizing* is True.
    max_power_mw : float
        Max charge / discharge rate in MW.  Ignored when *optimize_sizing*
        is True.
    efficiency : float
        Round-trip efficiency (0–1).  Applied symmetrically as
        ``sqrt(efficiency)`` on each side of the round trip.
    max_grid_mw : float
        Hard upper bound on power in MW (grid connection limit).
    optimize_sizing : bool
        When True the optimizer co-selects the optimal capacity (MWh) and
        power rating (MW) subject to the E:P ratio bounds.
    ep_ratio_min : float
        Minimum energy-to-power ratio in hours (default 2 h).
        Only applied when *optimize_sizing* is True.
    ep_ratio_max : float
        Maximum energy-to-power ratio in hours (default 4 h).
        Only applied when *optimize_sizing* is True.
    rfnbo_compliant : bool
        When True, charging is blocked during hours where the electricity
        price exceeds 20 EUR/MWh (RFNBO additionality proxy).
    initial_soc : float
        Initial state of charge as a fraction of capacity (0–1).

    Returns
    -------
    dict
        status              : solver termination condition string
        annual_revenue_eur  : total annual arbitrage revenue in EUR
        optimal_capacity_mwh: solved capacity (= input when optimize_sizing=False)
        optimal_power_mw    : solved power rating (= input when optimize_sizing=False)
        schedule            : dict of hourly arrays
                              (Hour, Price_EUR_MWh, Charge_MW, Discharge_MW,
                               SoC_MWh, Revenue_EUR)
        kpis                : dict of summary KPIs

    Raises
    ------
    ImportError  if Pyomo is not installed.
    """
    if not _PYOMO_OK:
        raise ImportError(
            "Pyomo is not installed.\n"
            "Install it with:  pip install pyomo\n"
            "Install HiGHS with:  pip install highspy"
        )

    prices = load_prices(price_bytes)
    T = len(prices)  # 8760
    sqrt_eff = math.sqrt(efficiency)

    if optimize_sizing:
        p_bound = max_grid_mw
        print(
            f"[optimizer] Building MILP | T={T} h | co-sizing mode | "
            f"E:P={ep_ratio_min}–{ep_ratio_max} h | grid_cap={max_grid_mw} MW | "
            f"eff={efficiency}"
        )
    else:
        p_bound = min(max_power_mw, max_grid_mw)
        print(
            f"[optimizer] Building MILP | T={T} h | cap={battery_capacity_mwh} MWh | "
            f"p_max={p_bound} MW | eff={efficiency}"
        )

    # ── Pyomo model ───────────────────────────────────────────────────────────
    m = ConcreteModel()
    m.T = Set(initialize=range(T))

    m.charge      = Var(m.T, domain=NonNegativeReals, bounds=(0, p_bound))
    m.discharge   = Var(m.T, domain=NonNegativeReals, bounds=(0, p_bound))
    m.soc         = Var(m.T, domain=NonNegativeReals)
    m.is_charging = Var(m.T, domain=Binary)

    if optimize_sizing:
        m.cap   = Var(domain=NonNegativeReals, bounds=(0.5, max_grid_mw * ep_ratio_max))
        m.power = Var(domain=NonNegativeReals, bounds=(0.5, max_grid_mw))
    else:
        cap   = battery_capacity_mwh
        power = p_bound

    # ── Objective ─────────────────────────────────────────────────────────────
    def _revenue(m):
        return sum(
            prices[t] * (m.discharge[t] * sqrt_eff - m.charge[t] / sqrt_eff)
            for t in m.T
        )

    m.obj = Objective(rule=_revenue, sense=maximize)

    # ── Constraints ───────────────────────────────────────────────────────────

    # 1. SoC upper bound
    def _soc_upper(m, t):
        return m.soc[t] <= (m.cap if optimize_sizing else cap)

    m.soc_upper = Constraint(m.T, rule=_soc_upper)

    # 2. Initial SoC
    _init_soc_frac = max(0.0, min(1.0, initial_soc))

    def _soc_init(m):
        return m.soc[0] == _init_soc_frac * (m.cap if optimize_sizing else cap)

    m.soc_init = Constraint(rule=_soc_init)

    # 3. SoC dynamics: soc[t] = soc[t-1] + charge[t-1]*η½ - discharge[t-1]/η½
    def _soc_dynamics(m, t):
        if t == 0:
            return Constraint.Skip
        return (
            m.soc[t]
            == m.soc[t - 1]
            + m.charge[t - 1] * sqrt_eff
            - m.discharge[t - 1] / sqrt_eff
        )

    m.soc_dynamics = Constraint(m.T, rule=_soc_dynamics)

    # 4. Terminal SoC ≥ 50 % (prevents end-of-year drain exploitation)
    def _soc_terminal(m):
        return m.soc[T - 1] >= 0.5 * (m.cap if optimize_sizing else cap)

    m.soc_terminal = Constraint(rule=_soc_terminal)

    # 5. Explicit power-limit constraints (needed when m.power is a Var)
    def _charge_power(m, t):
        return m.charge[t] <= (m.power if optimize_sizing else power)

    def _discharge_power(m, t):
        return m.discharge[t] <= (m.power if optimize_sizing else power)

    m.charge_power    = Constraint(m.T, rule=_charge_power)
    m.discharge_power = Constraint(m.T, rule=_discharge_power)

    # 6. Mutex: no simultaneous charge + discharge (Big-M linearisation)
    big_m = p_bound * 2

    def _mutex_charge(m, t):
        return m.charge[t] <= big_m * m.is_charging[t]

    def _mutex_discharge(m, t):
        return m.discharge[t] <= big_m * (1 - m.is_charging[t])

    m.mutex_charge    = Constraint(m.T, rule=_mutex_charge)
    m.mutex_discharge = Constraint(m.T, rule=_mutex_discharge)

    # 7. E:P ratio bounds (sizing mode only)
    if optimize_sizing:
        m.ep_min = Constraint(rule=lambda m: m.cap >= ep_ratio_min * m.power)
        m.ep_max = Constraint(rule=lambda m: m.cap <= ep_ratio_max * m.power)

    # 8. RFNBO additionality gate: block charging at price > 20 EUR/MWh
    if rfnbo_compliant:
        def _rfnbo_gate(m, t):
            if prices[t] > 20.0:
                return m.charge[t] == 0
            return Constraint.Skip

        m.rfnbo_gate = Constraint(m.T, rule=_rfnbo_gate)
        print("[optimizer] RFNBO gate active: charging blocked at price > 20 EUR/MWh")

    # ── Solve ──────────────────────────────────────────────────────────────────
    solver = SolverFactory("highs")
    if not solver.available():
        solver = SolverFactory("cbc")
        print("[optimizer] HiGHS not found — falling back to CBC")
    else:
        print("[optimizer] Solver: HiGHS")

    print("[optimizer] Solving MILP …")
    result = solver.solve(m, tee=False, options={"time_limit": 120})
    status = str(result.solver.termination_condition)
    print(f"[optimizer] Solver status: {status}")

    # ── Extract results ────────────────────────────────────────────────────────
    solved_cap   = float(value(m.cap))   if optimize_sizing else battery_capacity_mwh
    solved_power = float(value(m.power)) if optimize_sizing else p_bound

    charge_arr    = np.clip([value(m.charge[t])    for t in range(T)], 0, solved_power)
    discharge_arr = np.clip([value(m.discharge[t]) for t in range(T)], 0, solved_power)
    soc_arr       = np.clip([value(m.soc[t])       for t in range(T)], 0, None)

    charge_arr    = np.asarray(charge_arr,    dtype=float)
    discharge_arr = np.asarray(discharge_arr, dtype=float)
    soc_arr       = np.asarray(soc_arr,       dtype=float)

    rev_per_hour   = prices * (discharge_arr * sqrt_eff - charge_arr / sqrt_eff)
    annual_revenue = float(rev_per_hour.sum())

    total_charged    = float(charge_arr.sum())
    total_discharged = float(discharge_arr.sum())
    n_cycles         = total_discharged / max(solved_cap, 1e-6)

    print(
        f"[optimizer] Annual revenue: €{annual_revenue:,.0f} | "
        f"Capacity: {solved_cap:.1f} MWh | Power: {solved_power:.1f} MW | "
        f"E:P={solved_cap / max(solved_power, 1e-6):.1f} h | Cycles/yr: {n_cycles:.0f}"
    )

    return {
        "status":               status,
        "annual_revenue_eur":   annual_revenue,
        "optimal_capacity_mwh": solved_cap,
        "optimal_power_mw":     solved_power,
        "schedule": {
            "Hour":          list(range(1, T + 1)),
            "Price_EUR_MWh": prices.tolist(),
            "Charge_MW":     charge_arr.tolist(),
            "Discharge_MW":  discharge_arr.tolist(),
            "SoC_MWh":       soc_arr.tolist(),
            "Revenue_EUR":   rev_per_hour.tolist(),
        },
        "kpis": {
            "annual_revenue_eur":   round(annual_revenue, 2),
            "optimal_capacity_mwh": round(solved_cap, 2),
            "optimal_power_mw":     round(solved_power, 2),
            "ep_ratio_h":           round(solved_cap / max(solved_power, 1e-6), 2),
            "efficiency_pct":       round(float(efficiency) * 100, 1),
            "total_charged_mwh":    round(total_charged, 1),
            "total_discharged_mwh": round(total_discharged, 1),
            "annual_cycles":        round(n_cycles, 1),
            "avg_charge_price_eur_mwh": round(
                float((prices * charge_arr).sum()) / max(total_charged, 1e-6), 2
            ),
            "avg_discharge_price_eur_mwh": round(
                float((prices * discharge_arr).sum()) / max(total_discharged, 1e-6), 2
            ),
            "rfnbo_compliant": bool(rfnbo_compliant),
            "solver_status":   status,
        },
    }


def optimize_peak_shaving(
    load_profile: np.ndarray,
    battery_capacity_mwh: float = 8.0,
    max_power_mw: float = 2.0,
    efficiency: float = 0.92,
    initial_soc: float = 0.5,
) -> dict:
    """
    MILP peak-shaving: minimise the maximum grid import across the year.

    Given an industrial load profile, the battery discharges during demand
    peaks to cut the highest grid import and reduce demand charges.

    Objective
    ---------
    Minimise  ``peak``  subject to  ``peak >= load[t] + charge[t] - discharge[t]``
    for all hours t.

    Parameters
    ----------
    load_profile : np.ndarray
        Hourly electricity demand [MW], shape (T,).
    battery_capacity_mwh : float
        Battery energy capacity [MWh].
    max_power_mw : float
        Maximum charge / discharge rate [MW].
    efficiency : float
        Round-trip efficiency (0–1).
    initial_soc : float
        Initial SoC as a fraction of capacity.

    Returns
    -------
    dict with keys:
        status          : solver termination condition
        peak_without_mw : maximum grid import without BESS [MW]
        peak_with_mw    : optimised maximum grid import with BESS [MW]
        peak_reduction_pct : percentage reduction in peak demand
        schedule        : dict of hourly arrays (Load_MW, GridImport_MW,
                          Charge_MW, Discharge_MW, SoC_MWh)
    """
    if not _PYOMO_OK:
        raise ImportError("Pyomo is not installed. Install with: pip install pyomo highspy")

    T        = len(load_profile)
    sqrt_eff = math.sqrt(efficiency)
    cap      = battery_capacity_mwh
    power    = max_power_mw

    print(
        f"[peak_shaving] Building MILP | T={T} h | cap={cap} MWh | "
        f"p_max={power} MW | eff={efficiency}"
    )

    m   = ConcreteModel()
    m.T = Set(initialize=range(T))

    m.charge      = Var(m.T, domain=NonNegativeReals, bounds=(0, power))
    m.discharge   = Var(m.T, domain=NonNegativeReals, bounds=(0, power))
    m.soc         = Var(m.T, domain=NonNegativeReals, bounds=(0, cap))
    m.is_charging = Var(m.T, domain=Binary)
    m.peak        = Var(domain=NonNegativeReals)   # the peak grid import we minimise

    from pyomo.environ import minimize
    m.obj = Objective(expr=m.peak, sense=minimize)

    # Peak must be >= net grid import at every hour
    def _peak_floor(m, t):
        return m.peak >= load_profile[t] + m.charge[t] - m.discharge[t]
    m.peak_floor = Constraint(m.T, rule=_peak_floor)

    # SoC dynamics
    def _soc_dynamics(m, t):
        if t == 0:
            return m.soc[0] == initial_soc * cap
        return m.soc[t] == m.soc[t - 1] + m.charge[t - 1] * sqrt_eff - m.discharge[t - 1] / sqrt_eff
    m.soc_dynamics = Constraint(m.T, rule=_soc_dynamics)

    # Terminal SoC >= 50% (prevent end-of-year drain)
    m.soc_terminal = Constraint(rule=lambda m: m.soc[T - 1] >= 0.5 * cap)

    # Mutex: no simultaneous charge + discharge
    big_m = power * 2
    m.mutex_charge = Constraint(
        m.T, rule=lambda m, t: m.charge[t] <= big_m * m.is_charging[t]
    )
    m.mutex_discharge = Constraint(
        m.T, rule=lambda m, t: m.discharge[t] <= big_m * (1 - m.is_charging[t])
    )

    # Solve
    solver = SolverFactory("highs")
    if not solver.available():
        solver = SolverFactory("cbc")
        print("[peak_shaving] HiGHS not found — falling back to CBC")
    else:
        print("[peak_shaving] Solver: HiGHS")

    print("[peak_shaving] Solving MILP …")
    res    = solver.solve(m, tee=False, options={"time_limit": 120})
    status = str(res.solver.termination_condition)
    print(f"[peak_shaving] Status: {status}")

    charge_arr    = np.clip([value(m.charge[t])    for t in range(T)], 0, power)
    discharge_arr = np.clip([value(m.discharge[t]) for t in range(T)], 0, power)
    soc_arr       = np.clip([value(m.soc[t])       for t in range(T)], 0, cap)
    grid_import   = load_profile + np.asarray(charge_arr) - np.asarray(discharge_arr)

    peak_without = float(load_profile.max())
    peak_with    = float(grid_import.max())
    reduction    = (peak_without - peak_with) / peak_without * 100

    print(
        f"[peak_shaving] Peak without BESS: {peak_without:.2f} MW | "
        f"With BESS: {peak_with:.2f} MW | Reduction: {reduction:.1f}%"
    )

    return {
        "status":              status,
        "peak_without_mw":     round(peak_without, 3),
        "peak_with_mw":        round(peak_with, 3),
        "peak_reduction_pct":  round(reduction, 1),
        "schedule": {
            "Load_MW":       load_profile.tolist(),
            "GridImport_MW": grid_import.tolist(),
            "Charge_MW":     charge_arr,
            "Discharge_MW":  discharge_arr,
            "SoC_MWh":       soc_arr,
        },
    }
