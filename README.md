# BESS Arbitrage Optimizer

[![CI](https://github.com/parthivmehta1812/bess-arbitrage-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/parthivmehta1812/bess-arbitrage-optimizer/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python toolkit for BESS dispatch optimisation covering three use cases:

- **Arbitrage** — revenue-maximising charge/discharge schedule against hourly day-ahead prices (MILP, Pyomo + HiGHS)
- **Peak shaving** — minimise maximum grid import to cut industrial demand charges (MILP)
- **24h-ahead price forecasting** — gradient-boosted regression trees with calibrated 80% prediction intervals, evaluated against actual revenue capture

---

## Results

### Arbitrage — 2 MW / 8 MWh battery, 2023 German ENTSO-E prices

![Dispatch & Market Prices](docs/images/dispatch_snapshot.png)

![Monthly Revenue](docs/images/monthly_revenue.png)

| Parameter | Unconstrained | RFNBO Compliant |
|---|---|---|
| Annual Revenue | €289,981 | €194,709 |
| Annual Cycles | 539 | 211 |
| Avg Charge Price | 49.68 €/MWh | −2.91 €/MWh |
| Avg Discharge Price | 128.74 €/MWh | 117.05 €/MWh |
| RFNBO Revenue Penalty | — | €95,271 (32.9%) |
| Solver | HiGHS (optimal) | HiGHS (optimal) |

### Peak Shaving — synthetic industrial factory load

![Peak Shaving](docs/images/peak_shaving.png)

| Metric | Value |
|---|---|
| Peak demand without BESS | 2.30 MW |
| Peak demand with BESS | 1.61 MW |
| Peak reduction | **30.1%** |
| Battery | 2 MW / 8 MWh |

### Price Forecasting — Sep–Dec hold-out

![Price Forecasting vs Perfect Foresight](docs/images/forecast_vs_perfect.png)

GBM trained on Jan–Jul (hours 168–5 088), evaluated on Sep–Dec (hours 5 832–8 760).

| Metric | Value |
|---|---|
| MAE | 25.1 €/MWh |
| RMSE | 34.0 €/MWh |
| CRPS | 17.0 €/MWh |
| p90 empirical coverage | 80.3% (target 90%) |
| Revenue capture vs perfect foresight | 82.9% |

The p90 quantile is ~10 points too tight; the p10 lower bound is set too high (prices fall below it more often than the 10% target). See CQR section below.

---

## Forecast recalibration (CQR)

Conformal Quantile Regression adds a leakage-free post-hoc correction. Offsets learned on August (calibration set), applied to Sep–Dec (test set) — no test-period information used anywhere.

![Forecast Recalibration](docs/images/forecast_recalibration.png)

![Forecast Calibration](docs/images/forecast_calibration.png)

| Metric | (a) Uncalibrated | (b) Static CQR | (c) Rolling CQR |
|---|---|---|---|
| p10 coverage (target 10%) | 22.7% | 23.3% | 13.8% |
| p90 coverage (target 90%) | 80.3% | 85.4% | 89.5% |
| 80% interval coverage (target 80%) | 57.6% | 62.1% | 75.7% |
| Mean interval width (EUR/MWh) | 49.60 | 53.94 | 74.82 |
| CRPS (EUR/MWh) | 16.979 | 16.810 | 16.297 |

Static CQR offsets: +0.81 / +5.15 EUR/MWh for p10/p90. Rolling CQR reaches 89.5% p90 coverage (0.5 pp short of target) and reduces CRPS from 16.98 to 16.30 EUR/MWh at the cost of wider intervals. Numbers as measured; nothing tuned on the test period.

Run [`examples/04_recalibration.py`](examples/04_recalibration.py) to reproduce.

---

## Walk-forward backtest

Expanding-window monthly re-training, March–December 2023. GBM trained on all hours before each test month; persistence baseline is price[*t*−24]. All three strategies (perfect foresight, GBM p50, persistence) are **evaluated on actual prices**.

![Walk-forward backtest](docs/images/backtest_monthly.png)

| | MAE | RMSE | p90 cov% | 80% cov% | CRPS | MAE pers | RMSE pers |
|---|---|---|---|---|---|---|---|
| **Overall (Mar–Dec)** | **22.8** | **32.1** | **84.9** | **61.8** | **15.92** | **24.5** | **34.8** |

| | PF revenue | GBM revenue | Pers revenue | Capture GBM | Capture pers |
|---|---|---|---|---|---|
| **Overall (Mar–Dec)** | **€261,695** | **€230,592** | **€229,535** | **88.1%** | **87.7%** |

**Diebold–Mariano** (GBM p50 vs persistence, *n* = 306 days, one-sided H₁: GBM lower loss):

| | Abs loss | Sq loss |
|---|---|---|
| Mean loss — persistence | 24.475 €/MWh | 1 213.8 (€/MWh)² |
| Mean loss — GBM p50 | 22.815 €/MWh | 1 032.3 (€/MWh)² |
| HLN statistic | 2.160 | 2.642 |
| *p*-value (HLN, *t*₃₀₅) | 0.016 | 0.004 |

GBM is significantly more accurate than persistence (DM *p* = 0.016 abs, 0.004 sq) and marginally wins on revenue capture overall (88.1% vs 87.7%, 7 of 10 months). March is the weakest month (79.5% GBM vs 89.2% persistence) due to sparse training data (~1 250 samples). 80% interval coverage averages 61.8% and p90 coverage 84.9%, both below their targets.

**Limitations:** monthly (not daily) re-training; single year of data (2023).

Run [`examples/05_backtest.py`](examples/05_backtest.py) to reproduce.

---

## Robustness and fallback

Tests how plausibility checks, imputation, and a confidence fallback protect revenue under faulty price inputs. Single seed (42).

**Fault injection — scenario S1** (Mar–Dec only): 376 NaN hours (5.1%), 255 spike hours (3.5%), 201 stuck-value hours (2.7%); 762 total imputed (10.4%).

**Strategies:** A = GBM no checks · B = GBM + checks + imputation + fallback · C = persistence + checks · D = perfect foresight.

![Robustness evaluation](docs/images/robustness.png)

| Scenario | A cap% | B cap% | B fallback% | C cap% |
|---|---|---|---|---|
| S0 — clean | 88.1 | 88.1 | 8.5 | 87.2 |
| S1 — faulty | 83.5 | 85.6 | 43.1 | 85.3 |

On clean data the three strategies are close. On faulty data the robustness layer (B) recovers 2.1 pp over raw GBM (A) by switching ~43% of days to cleaned persistence; cleaned persistence alone (C, 85.3%) is comparable. The S0→S1 degradation for raw GBM (88.1% → 83.5%) is modest compared to the original results — much of the previously reported drop was artefact of the look-ahead leak.

**Limitations:** single fault seed; monthly planning horizon; no intra-month re-forecasting.

Run [`examples/06_robustness.py`](examples/06_robustness.py) to reproduce.

---

## Correction (Oct 2026)

The rolling-feature window in `_make_features` originally used `prices[t-24:t]`, which included same-day hours not yet known at bid time (noon on day D−1). The fix uses `prices[day_start-24:day_start]` — the previous complete day only. `model_p50` was also corrected from squared-error loss (mean) to quantile loss at α=0.50 (true median), and per-hour quantile ordering (p10 ≤ p50 ≤ p90) is now enforced. After the fix, MAE rose from 21.3 to 22.8 EUR/MWh but GBM revenue capture improved from 80.9% to 88.1%, because the corrected model no longer learns spurious intra-day correlations that misled the dispatch optimiser.

---

## Features

| Feature | Details |
|---|---|
| **Arbitrage MILP** | Maximise day-ahead revenue; charge low, discharge high |
| **Peak-shaving MILP** | Minimise max grid import; cuts industrial demand charges |
| **Co-sizing mode** | Jointly optimise capacity, power, and schedule under an E:P ratio window |
| **RFNBO compliance gate** | Block charging at prices > 20 EUR/MWh (additionality proxy) |
| **Round-trip efficiency** | Symmetric √η split across charge and discharge legs |
| **Price forecaster** | GBM with cyclic calendar encoding, lagged prices, 80% prediction intervals; p50 is a true median (quantile loss α=0.50) |
| **Forecast calibration** | Reliability diagram and CRPS to evaluate probabilistic forecast quality |
| **Forecast-driven dispatch** | Run MILP on forecast prices; evaluate revenue capture vs perfect foresight |

---

## Problem Formulation

### Arbitrage dispatch

**Decision variables (per hour *t*):**

| Variable | Domain | Meaning |
|---|---|---|
| `charge[t]` | ℝ≥0 | MW drawn from grid |
| `discharge[t]` | ℝ≥0 | MW exported to grid |
| `soc[t]` | ℝ≥0 | MWh stored (State of Charge) |
| `is_charging[t]` | {0, 1} | Mutex binary: 1 = charging |

$$\max \sum_{t=1}^{8760} \lambda_t \left( \text{discharge}_t \cdot \sqrt{\eta} - \frac{\text{charge}_t}{\sqrt{\eta}} \right)$$

### Peak shaving

$$\min \; \text{peak} \quad \text{subject to} \quad \text{peak} \geq D_t + \text{charge}_t - \text{discharge}_t \quad \forall t$$

### Key constraints (both modes)

```
soc[t]  =  soc[t-1]  +  charge[t-1]·√η  −  discharge[t-1]/√η
0       ≤  soc[t]    ≤  cap
soc[T]  ≥  0.5 · cap
charge[t]    ≤  M · is_charging[t]          (mutex — Big-M)
discharge[t] ≤  M · (1 − is_charging[t])
```

---

## Installation

```bash
git clone https://github.com/parthivmehta1812/bess-arbitrage-optimizer.git
cd bess-arbitrage-optimizer
pip install -e ".[dev]"
pip install highspy   # recommended solver
```

---

## Quick Start

### Arbitrage

```python
from bess_arbitrage import optimize_bess_arbitrage

result = optimize_bess_arbitrage(battery_capacity_mwh=8.0, max_power_mw=2.0, efficiency=0.92)
print(f"Annual revenue : €{result['kpis']['annual_revenue_eur']:,.0f}")
print(f"Annual cycles  : {result['kpis']['annual_cycles']}")
```

### Peak shaving

```python
from bess_arbitrage import optimize_peak_shaving, make_industrial_load

load = make_industrial_load(peak_mw=2.2, base_mw=0.4)
result = optimize_peak_shaving(load, battery_capacity_mwh=8.0, max_power_mw=2.0)
print(f"Peak reduction : {result['peak_reduction_pct']:.1f}%")
```

### Price forecasting with calibration

```python
from bess_arbitrage import train_price_forecaster, forecast_prices
import numpy as np

prices = ...  # np.ndarray, 8760 hourly prices
models = train_price_forecaster(prices, train_end_h=5832)
point, lower, upper = forecast_prices(models, prices, start_h=5832, end_h=8760)
print(f"p90 coverage: {np.mean(prices[5832:] <= upper) * 100:.1f}%  (target 90%)")
```

See [`examples/`](examples/) for complete scripts.

---

## API Reference

### `optimize_bess_arbitrage(...) → dict`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `price_bytes` | `bytes \| None` | `None` | Raw CSV/XLSX bytes; `None` uses bundled 2023 DE data |
| `battery_capacity_mwh` | `float` | `8.0` | Fixed capacity in MWh |
| `max_power_mw` | `float` | `2.0` | Fixed power rating in MW |
| `efficiency` | `float` | `0.92` | Round-trip efficiency (0–1) |
| `optimize_sizing` | `bool` | `False` | Co-optimise capacity and power |
| `rfnbo_compliant` | `bool` | `False` | Block charging above 20 EUR/MWh |
| `initial_soc` | `float` | `0.5` | Initial SoC as fraction of capacity |

### `optimize_peak_shaving(load_profile, ...) → dict`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `load_profile` | `np.ndarray` | — | Hourly demand [MW], shape (T,) |
| `battery_capacity_mwh` | `float` | `8.0` | Battery capacity in MWh |
| `max_power_mw` | `float` | `2.0` | Max charge/discharge rate in MW |
| `efficiency` | `float` | `0.92` | Round-trip efficiency (0–1) |

**Returns** `peak_without_mw`, `peak_with_mw`, `peak_reduction_pct`, and hourly `schedule`.

### `train_price_forecaster(prices, train_end_h, n_estimators) → tuple`

Returns `(model_p50, model_p10, model_p90)` — three fitted scikit-learn GBM estimators. `model_p50` uses quantile loss at α=0.50 (true median).

### `forecast_prices(models, prices, start_h, end_h) → tuple`

Returns `(point, lower, upper)` — p50, p10, p90 forecasts as `np.ndarray`. Quantile ordering (p10 ≤ p50 ≤ p90) is enforced per hour.

---

## Running Tests

```bash
pytest tests/ -v --cov=bess_arbitrage
```

---

## Price Data

Bundled 2023 German ENTSO-E dataset (8,760 hourly values, min −129.8, max +583.4 EUR/MWh). For other years or markets, download from [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) and pass via `price_bytes`.

---

## License

MIT — see [LICENSE](LICENSE).
