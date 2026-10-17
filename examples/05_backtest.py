"""
Walk-forward backtest — monthly re-training, GBM vs persistence.

Data (2023 DE ENTSO-E prices, 8 760 h, no look-ahead anywhere)
---------------------------------------------------------------
  Test months  : March … December (10 months)
  Train window : expanding — hours [168, start_of_month) for each test month
  Persistence  : price[t-24]  (same hour yesterday)
  BESS         : 8 MWh / 2 MW / η=0.92 / initial SoC 0.5

Output
------
  Monthly + overall table (forecast quality + economic)
  Diebold–Mariano significance test (GBM p50 vs persistence)
  docs/images/backtest_monthly.png

NOTE: This script runs ~30 MILP optimisations (3 per test month) and
re-trains 10 GBM models.  Expect roughly 20–60 minutes wall time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bess_arbitrage.loader import load_prices
from bess_arbitrage.forecaster import train_price_forecaster, forecast_prices
from bess_arbitrage.backtest import (
    walk_forward_backtest,
    diebold_mariano,
    _MONTH_NAMES,
    _MONTH_STARTS,
)

# ── Load prices ───────────────────────────────────────────────────────────────
print("Loading 2023 DE price data …")
prices = load_prices()

# ── Walk-forward backtest ─────────────────────────────────────────────────────
print("\nStarting walk-forward backtest (March–December) …")
df = walk_forward_backtest(
    prices,
    first_test_month    = 3,
    battery_capacity_mwh= 8.0,
    max_power_mw        = 2.0,
    efficiency          = 0.92,
    initial_soc         = 0.5,
    n_estimators        = 250,
)

# ── Print results tables ──────────────────────────────────────────────────────
W1, W2 = 9, 10

# Forecast-quality table
print()
print("=" * 85)
print("  Walk-forward backtest — forecast quality (GBM p50 vs actuals)")
print("=" * 85)
hdr = (
    f"{'Month':<{W1}}"
    f"{'N hours':>{W2}}"
    f"{'MAE':>{W2}}"
    f"{'RMSE':>{W2}}"
    f"{'p90 cov%':>{W2}}"
    f"{'80% cov%':>{W2}}"
    f"{'CRPS':>{W2}}"
    f"{'MAE pers':>{W2}}"
    f"{'RMSE pers':>{W2}}"
)
print(hdr)
print("-" * len(hdr))
for _, row in df.iterrows():
    is_overall = row["month"] == "Overall"
    sep = "=" if is_overall else " "
    if is_overall:
        print("-" * len(hdr))
    print(
        f"{row['month']:<{W1}}"
        f"{int(row['n_hours']):>{W2}}"
        f"{row['mae']:>{W2}.1f}"
        f"{row['rmse']:>{W2}.1f}"
        f"{row['p90_cov_pct']:>{W2}.1f}"
        f"{row['int80_cov_pct']:>{W2}.1f}"
        f"{row['crps']:>{W2}.2f}"
        f"{row['mae_pers']:>{W2}.1f}"
        f"{row['rmse_pers']:>{W2}.1f}"
    )
print("=" * len(hdr))

# Economic table
print()
print("=" * 90)
print("  Walk-forward backtest — economic results (revenue on actual prices)")
print("=" * 90)
hdr2 = (
    f"{'Month':<{W1}}"
    f"{'PF revenue':>{W2+2}}"
    f"{'GBM revenue':>{W2+2}}"
    f"{'Pers revenue':>{W2+2}}"
    f"{'Capture GBM':>{W2+2}}"
    f"{'Capture pers':>{W2+2}}"
)
print(hdr2)
print("-" * len(hdr2))
for _, row in df.iterrows():
    if row["month"] == "Overall":
        print("-" * len(hdr2))
    print(
        f"{row['month']:<{W1}}"
        f"€{row['revenue_perfect_eur']:>{W2+1},.0f}"
        f"€{row['revenue_gbm_eur']:>{W2+1},.0f}"
        f"€{row['revenue_pers_eur']:>{W2+1},.0f}"
        f"{row['capture_gbm_pct']:>{W2+2}.1f}%"
        f"{row['capture_pers_pct']:>{W2+2}.1f}%"
    )
print("=" * len(hdr2))

# ── Diebold–Mariano test (pooled test period, Mar–Dec) ───────────────────────
print("\nComputing Diebold–Mariano test (GBM p50 vs persistence, Mar–Dec) …")

first_test = 2   # 0-indexed: March
h_test_start = _MONTH_STARTS[first_test]   # 1416
h_test_end   = _MONTH_STARTS[12]           # 8760

# Re-train each month's GBM to pool errors — re-use the same expanding window
e_gbm_all  = []
e_pers_all = []

for m_idx in range(first_test, 12):
    h_start = _MONTH_STARTS[m_idx]
    h_end   = _MONTH_STARTS[m_idx + 1]
    actual  = prices[h_start:h_end]

    # GBM forecast (already computed during the backtest, but re-computed here
    # to keep the DM section self-contained and avoid storing intermediate arrays)
    models = train_price_forecaster(prices, train_end_h=h_start)
    p50, _, _ = forecast_prices(models, prices, h_start, h_end)

    pers = prices[h_start - 24 : h_end - 24]

    e_gbm_all.append(actual - p50)
    e_pers_all.append(actual - pers)

e_gbm  = np.concatenate(e_gbm_all)
e_pers = np.concatenate(e_pers_all)

dm_abs = diebold_mariano(e_pers, e_gbm, loss="abs")
dm_sq  = diebold_mariano(e_pers, e_gbm, loss="sq")

print()
print("=" * 72)
print("  Diebold–Mariano test: H1 = GBM has lower expected loss than persistence")
print("=" * 72)
print(f"  Test period: Mar–Dec 2023  |  n_days = {dm_abs['n_days']}")
print(f"  {'Metric':<24}  {'Abs loss':>12}  {'Sq loss':>12}")
print(f"  {'-'*24}  {'-'*12}  {'-'*12}")
print(f"  {'Mean loss (persistence)':<24}  {dm_abs['loss_mean_e1']:>12.3f}  {dm_sq['loss_mean_e1']:>12.3f}")
print(f"  {'Mean loss (GBM p50)':<24}  {dm_abs['loss_mean_e2']:>12.3f}  {dm_sq['loss_mean_e2']:>12.3f}")
print(f"  {'DM statistic':<24}  {dm_abs['dm_stat']:>12.3f}  {dm_sq['dm_stat']:>12.3f}")
print(f"  {'p-value (normal)':<24}  {dm_abs['p_normal']:>12.4f}  {dm_sq['p_normal']:>12.4f}")
print(f"  {'HLN statistic':<24}  {dm_abs['hln_stat']:>12.3f}  {dm_sq['hln_stat']:>12.3f}")
print(f"  {'p-value (HLN, t-dist)':<24}  {dm_abs['p_hln']:>12.4f}  {dm_sq['p_hln']:>12.4f}")
print("=" * 72)

# ── Plot ──────────────────────────────────────────────────────────────────────
OUT_DIR = ROOT / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    C = {
        "brand":        "#4C09E7",
        "brandLight":   "#A485F6",
        "brandLighter": "#D9CCFB",
        "success":      "#22C55E",
        "error":        "#EF4444",
        "gray900":      "#18122B",
        "gray600":      "#574F7D",
        "gray200":      "#DCD7EC",
        "white":        "#ffffff",
        "plotBg":       "#faf9fe",
        "orange":       "#F97316",
    }
    AXIS_STYLE = dict(
        gridcolor=C["gray200"], linecolor=C["gray200"], zerolinecolor=C["gray200"]
    )

    test_rows = df[df["month"] != "Overall"].copy()
    months    = test_rows["month"].tolist()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Monthly revenue capture (%)", "Monthly MAE (€/MWh)"),
        horizontal_spacing=0.12,
    )

    bar_w = 0.35
    x_pos = list(range(len(months)))

    # Revenue capture bars
    fig.add_trace(go.Bar(
        name="GBM p50",
        x=[x - bar_w / 2 for x in x_pos],
        y=test_rows["capture_gbm_pct"].tolist(),
        width=bar_w,
        marker=dict(color=C["brand"], opacity=0.85),
        text=[f"{v:.1f}%" for v in test_rows["capture_gbm_pct"]],
        textposition="outside",
        textfont=dict(size=9, color=C["gray600"]),
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        name="Persistence",
        x=[x + bar_w / 2 for x in x_pos],
        y=test_rows["capture_pers_pct"].tolist(),
        width=bar_w,
        marker=dict(color=C["orange"], opacity=0.85),
        text=[f"{v:.1f}%" for v in test_rows["capture_pers_pct"]],
        textposition="outside",
        textfont=dict(size=9, color=C["gray600"]),
    ), row=1, col=1)

    # 100 % reference line
    fig.add_trace(go.Scatter(
        name="Perfect foresight (100%)",
        x=[-0.6, len(months) - 0.4],
        y=[100, 100],
        mode="lines",
        line=dict(color=C["success"], dash="dash", width=1.5),
        showlegend=True,
    ), row=1, col=1)

    # MAE bars
    fig.add_trace(go.Bar(
        name="GBM p50",
        x=[x - bar_w / 2 for x in x_pos],
        y=test_rows["mae"].tolist(),
        width=bar_w,
        marker=dict(color=C["brand"], opacity=0.85),
        text=[f"{v:.1f}" for v in test_rows["mae"]],
        textposition="outside",
        textfont=dict(size=9, color=C["gray600"]),
        showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        name="Persistence",
        x=[x + bar_w / 2 for x in x_pos],
        y=test_rows["mae_pers"].tolist(),
        width=bar_w,
        marker=dict(color=C["orange"], opacity=0.85),
        text=[f"{v:.1f}" for v in test_rows["mae_pers"]],
        textposition="outside",
        textfont=dict(size=9, color=C["gray600"]),
        showlegend=False,
    ), row=1, col=2)

    overall_row = df[df["month"] == "Overall"].iloc[0]
    fig.update_layout(
        paper_bgcolor=C["white"],
        plot_bgcolor=C["plotBg"],
        font=dict(family="Inter, Arial, sans-serif", size=12, color=C["gray600"]),
        title=dict(
            text=(
                f"Walk-Forward Backtest (Mar–Dec 2023) — "
                f"Overall capture: GBM {overall_row['capture_gbm_pct']:.1f}%  "
                f"Pers {overall_row['capture_pers_pct']:.1f}%  |  "
                f"MAE: GBM {overall_row['mae']:.1f}  Pers {overall_row['mae_pers']:.1f} €/MWh"
            ),
            font=dict(size=12, color=C["gray900"]), x=0.01,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)", bordercolor=C["gray200"], borderwidth=1
        ),
        margin=dict(t=70, r=40, b=65, l=70),
        height=460,
        width=1200,
        barmode="group",
    )
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    fig.update_xaxes(
        tickvals=x_pos, ticktext=months, tickangle=-30, row=1, col=1
    )
    fig.update_xaxes(
        tickvals=x_pos, ticktext=months, tickangle=-30, row=1, col=2
    )
    fig.update_yaxes(title_text="Revenue capture (%)", row=1, col=1)
    fig.update_yaxes(title_text="MAE (€/MWh)", row=1, col=2)

    plot_path = OUT_DIR / "backtest_monthly.png"
    fig.write_image(str(plot_path), format="png", width=1200, height=460, scale=2)
    print(f"\nSaved: {plot_path}")

except ImportError as exc:
    print(f"\n[plot] Skipped (missing dependency: {exc})")
