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

| Parameter | Unconstrained | RFNBO Compliant |
|---|---|---|
| Annual Revenue | €289,981 | €194,709 |
| Annual Cycles | 539 | 211 |
| Avg Charge Price | 49.68 €/MWh | −2.91 €/MWh |
| Avg Discharge Price | 128.74 €/MWh | 117.05 €/MWh |
| RFNBO Revenue Penalty | — | €95,271 (32.9%) |
| Solver | HiGHS (optimal) | HiGHS (optimal) |

### Peak Shaving — synthetic industrial factory load

| Metric | Value |
|---|---|
| Peak demand without BESS | 2.30 MW |
| Peak demand with BESS | 1.61 MW |
| Peak reduction | **30.1%** |
| Battery | 2 MW / 8 MWh |

### Price Forecasting — Sep–Dec hold-out

| Metric | Value |
|---|---|
| MAE | 21.1 €/MWh |
| RMSE | 28.1 €/MWh |
| CRPS | 14.1 €/MWh |
| p90 empirical coverage | 82.8% (target 90%) |
| Revenue capture vs perfect foresight | 77.0% |

> **Calibration note:** the p90 quantile covers 82.8% of hold-out hours against a 90% target, so the upper bound is about 7 points too tight. The intervals are reported as measured and have not been recalibrated; see the reliability diagram below.

---

## Forecast recalibration (CQR)

Conformal Quantile Regression (CQR) adds a leakage-free post-hoc correction to the GBM quantile forecasts.

### Data split

| Segment | Hours | Period | Purpose |
|---|---|---|---|
| Train | 168 – 5 088 | Jan–Jul | Fit GBM models |
| Calibration | 5 088 – 5 832 | August | Learn CQR offsets (never used to fit anything else) |
| Test | 5 832 – 8 760 | Sep–Dec | Evaluate all variants |

The calibration set is kept strictly separate from training and test. Offsets are learned on August and applied to Sep–Dec — no information from the test period is used anywhere.

### Results — test period Sep–Dec 2023

| Metric | (a) Uncalibrated | (b) Static CQR | (c) Rolling CQR |
|---|---|---|---|
| p10 coverage (target 10%) | 18.6% | 19.5% | 12.4% |
| p90 coverage (target 90%) | 82.2% | 84.3% | 89.3% |
| 80% interval coverage (target 80%) | 63.6% | 64.8% | 76.9% |
| Mean interval width (EUR/MWh) | 45.51 | 46.91 | 64.46 |
| Pinball loss p10 (EUR/MWh) | 5.268 | 5.331 | 4.989 |
| Pinball loss p50 (EUR/MWh) | 10.618 | 10.563 | 10.618 |
| Pinball loss p90 (EUR/MWh) | 5.235 | 5.095 | 4.886 |
| CRPS (EUR/MWh) | 14.080 | 13.993 | 13.661 |

Numbers are as measured; nothing was tuned on the test period.

**Interpretation:** Static CQR (August offsets, +0.72 / +1.97 EUR/MWh for p10/p90) closes only 2 of the 8 percentage-point gap in p90 coverage. The 30-day rolling variant reaches 89.3% p90 coverage — close to but still 0.7 points short of the 90% target — while also improving CRPS from 14.08 to 13.66 EUR/MWh. The p10 quantile remains above the 10% target in all variants; the model's lower bound is systematically too pessimistic and CQR only partially corrects it. Interval widths widen with calibration (especially rolling), which is the expected cost of better coverage.

![Forecast Recalibration](docs/images/forecast_recalibration.png)

Run [`examples/04_recalibration.py`](examples/04_recalibration.py) to reproduce these results.

---

## Walk-forward backtest

Expanding-window monthly re-training over March–December 2023. For each test month *m*:

1. Train the GBM forecaster on hours 168 … *start\_of(m)* — strictly no look-ahead.
2. Produce 24h-ahead p10/p50/p90 forecasts for month *m*.
3. Naive persistence baseline: price\[*t*−24\] (same hour yesterday).
4. Run three MILP optimisations (8 MWh / 2 MW / η = 0.92 / SoC₀ = 0.5), all **evaluated on actual prices**: perfect foresight, GBM p50, persistence.

### Data split

| Segment | Hours | Period | Purpose |
|---|---|---|---|
| Train (expanding) | 168 → *start\_of(m)* | Jan–*prev. month* | Fit GBM models |
| Test | *start\_of(m)* → *end\_of(m)* | Month *m* | Evaluate forecast + dispatch |
| First test month | 1 416 | March | GBM trained on Jan–Feb only |
| Last test month | 8 016 | December | GBM trained on Jan–Nov |

### Forecast quality (GBM p50 vs actuals) — Mar–Dec 2023

All metrics in €/MWh except coverage (%). Overall row weighted by month length.

| Month | MAE | RMSE | p90 cov% | 80% cov% | CRPS | MAE pers | RMSE pers |
|---|---|---|---|---|---|---|---|
| Mar | 33.8 | 44.5 | 88.2 | 58.3 | 23.34 | 26.7 | 40.8 |
| Apr | 25.9 | 34.6 | 88.5 | 70.7 | 16.25 | 24.8 | 35.3 |
| May | 26.3 | 34.2 | 78.2 | 57.1 | 17.37 | 26.5 | 37.6 |
| Jun | 19.1 | 23.7 | 88.2 | 81.1 | 11.39 | 20.6 | 29.1 |
| Jul | 14.3 | 19.6 | 90.7 | 76.9 | 9.25 | 20.8 | 30.2 |
| Aug | 11.5 | 16.3 | 87.5 | 78.2 | 7.77 | 19.2 | 29.1 |
| Sep | 18.0 | 24.9 | 84.7 | 65.6 | 12.14 | 24.0 | 33.6 |
| Oct | 22.3 | 29.4 | 82.1 | 62.0 | 14.90 | 31.3 | 41.6 |
| Nov | 20.4 | 27.3 | 77.6 | 64.3 | 13.34 | 25.2 | 33.5 |
| Dec | 20.9 | 27.7 | 91.7 | 63.4 | 12.95 | 25.7 | 34.7 |
| **Overall** | **21.3** | **29.3** | **85.8** | **67.7** | **13.88** | **24.5** | **34.8** |

### Economic results (revenue evaluated on actual prices) — Mar–Dec 2023

| Month | PF revenue | GBM revenue | Pers revenue | Capture GBM | Capture pers |
|---|---|---|---|---|---|
| Mar | €27,353 | €12,107 | €24,408 | 44.3% | 89.2% |
| Apr | €29,514 | €23,754 | €27,537 | 80.5% | 93.3% |
| May | €33,879 | €29,480 | €30,310 | 87.0% | 89.5% |
| Jun | €28,047 | €24,836 | €24,906 | 88.6% | 88.8% |
| Jul | €30,119 | €27,314 | €26,709 | 90.7% | 88.7% |
| Aug | €28,255 | €26,536 | €25,715 | 93.9% | 91.0% |
| Sep | €28,287 | €25,802 | €25,303 | 91.2% | 89.5% |
| Oct | €26,223 | €21,230 | €21,929 | 81.0% | 83.6% |
| Nov | €16,939 | €11,888 | €13,291 | 70.2% | 78.5% |
| Dec | €13,078 | €8,653 | €9,426 | 66.2% | 72.1% |
| **Overall** | **€261,695** | **€211,598** | **€229,535** | **80.9%** | **87.7%** |

### Diebold–Mariano test (GBM p50 vs persistence, Mar–Dec, *n* = 306 days)

H₁: GBM has lower expected loss than persistence (one-sided). Daily loss differentials, Newey–West lag = 1, HLN small-sample correction.

| Metric | Abs loss | Sq loss |
|---|---|---|
| Mean loss — persistence | 24.475 €/MWh | 1 213.8 (€/MWh)² |
| Mean loss — GBM p50 | 21.265 €/MWh | 857.4 (€/MWh)² |
| DM statistic | 3.741 | 4.713 |
| *p*-value (normal) | 0.0001 | < 0.0001 |
| HLN statistic | 3.735 | 4.706 |
| *p*-value (HLN, *t*₃₀₅) | 0.0001 | < 0.0001 |

### Plot

![Walk-forward backtest](docs/images/backtest_monthly.png)

Run [`examples/05_backtest.py`](examples/05_backtest.py) to reproduce.

### Interpretation

GBM is highly significantly more accurate than persistence in point-forecast terms — DM *p* < 0.001 for both absolute and squared loss, with or without the HLN correction. That statistical advantage does **not** translate to better revenue capture: over Mar–Dec persistence (87.7%) outperforms GBM p50 (80.9%). GBM wins only in Jul (90.7% vs 88.7%) and Aug (93.9% vs 91.0%) — the months with the most training data. March is especially poor for GBM (44.3% vs 89.2%): with only ~1 250 training samples the model mis-forecasts the seasonal price structure enough to drive a badly timed dispatch schedule. GBM's 80% interval coverage averages 67.7%, well short of the 80% target; p90 coverage (85.8%) also falls below 90%.

**Limitations:** monthly (not daily) re-training; single year of data (2023).

---

## Plots

### Arbitrage dispatch — Week 1
![Dispatch & Market Prices](docs/images/dispatch_snapshot.png)

### Monthly arbitrage revenue
![Monthly Revenue](docs/images/monthly_revenue.png)

### Peak shaving — industrial load vs grid import
![Peak Shaving](docs/images/peak_shaving.png)

### Forecast accuracy and revenue capture
![Price Forecasting vs Perfect Foresight](docs/images/forecast_vs_perfect.png)

### Forecast calibration — reliability diagram + CRPS
![Forecast Calibration](docs/images/forecast_calibration.png)

---

## Features

| Feature | Details |
|---|---|
| **Arbitrage MILP** | Maximise day-ahead revenue; charge low, discharge high |
| **Peak-shaving MILP** | Minimise max grid import; cuts industrial demand charges |
| **Co-sizing mode** | Jointly optimise capacity, power, and schedule under an E:P ratio window |
| **RFNBO compliance gate** | Block charging at prices > 20 EUR/MWh (additionality proxy) |
| **Round-trip efficiency** | Symmetric √η split across charge and discharge legs |
| **Price forecaster** | GBM with cyclic calendar encoding, lagged prices, 80% prediction intervals |
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

**Additional variable:**

| Variable | Domain | Meaning |
|---|---|---|
| `peak` | ℝ≥0 | Maximum grid import across all hours |

$$\min \; \text{peak} \quad \text{subject to} \quad \text{peak} \geq D_t + \text{charge}_t - \text{discharge}_t \quad \forall t$$

where $D_t$ is the industrial electricity demand at hour *t*.

### Key constraints (both modes)

```
soc[t]  =  soc[t-1]  +  charge[t-1]·√η  −  discharge[t-1]/√η     (dynamics)
0       ≤  soc[t]    ≤  cap                                         (capacity)
soc[0]  =  soc₀ · cap                                              (initial SoC)
soc[T]  ≥  0.5 · cap                                               (terminal SoC)
charge[t]    ≤  M · is_charging[t]                                  (mutex — Big-M)
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

result = optimize_bess_arbitrage(
    battery_capacity_mwh=8.0,
    max_power_mw=2.0,
    efficiency=0.92,
)
print(f"Annual revenue : €{result['kpis']['annual_revenue_eur']:,.0f}")
print(f"Annual cycles  : {result['kpis']['annual_cycles']}")
```

### Peak shaving

```python
from bess_arbitrage import optimize_peak_shaving, make_industrial_load

load = make_industrial_load(peak_mw=2.2, base_mw=0.4)
result = optimize_peak_shaving(load, battery_capacity_mwh=8.0, max_power_mw=2.0)

print(f"Peak without BESS : {result['peak_without_mw']:.2f} MW")
print(f"Peak with BESS    : {result['peak_with_mw']:.2f} MW")
print(f"Reduction         : {result['peak_reduction_pct']:.1f}%")
```

### Price forecasting with calibration

```python
from bess_arbitrage import train_price_forecaster, forecast_prices
import numpy as np

prices = ...  # np.ndarray, 8760 hourly prices

models = train_price_forecaster(prices, train_end_h=5832)
point, lower, upper = forecast_prices(models, prices, start_h=5832, end_h=8760)

# Empirical calibration check
actuals = prices[5832:8760]
print(f"p90 coverage: {np.mean(actuals <= upper) * 100:.1f}%  (target 90%)")
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

### `make_industrial_load(peak_mw, base_mw, seed) → np.ndarray`

Generates a synthetic 8,760-hour industrial load profile with day/night, weekday/weekend, and Gaussian noise.

### `train_price_forecaster(prices, train_end_h, n_estimators) → tuple`

Returns `(model_p50, model_p10, model_p90)` — three fitted scikit-learn GBM estimators.

### `forecast_prices(models, prices, start_h, end_h) → tuple`

Returns `(point, lower, upper)` — p50, p10, p90 forecasts as `np.ndarray`.

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
