"""
Generate README plots from a REAL optimizer run.
Matches the BESSResultsPage.jsx frontend exactly — same colors, chart types, layout.

Requirements:
    pip install pyomo highspy pandas numpy plotly kaleido

Usage:
    python scripts/generate_plots.py
"""

import sys
import json
from pathlib import Path

# Allow running from repo root or scripts/ folder
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bess_arbitrage import optimize_bess_arbitrage

OUT_DIR = ROOT / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Brand palette — exact match to BESSResultsPage.jsx ───────────────────────
C = {
    "brand":        "#4C09E7",
    "brandDeep":    "#2D156E",
    "brandLight":   "#A485F6",
    "brandLighter": "#D9CCFB",
    "success":      "#22C55E",
    "successDark":  "#16A34A",
    "error":        "#EF4444",
    "gray900":      "#18122B",
    "gray600":      "#574F7D",
    "gray500":      "#6E6599",
    "gray200":      "#DCD7EC",
    "white":        "#ffffff",
    "plotBg":       "#faf9fe",
}

# ── Shared Plotly layout — mirrors PLOT_LAYOUT in BESSResultsPage.jsx ─────────
AXIS_STYLE = dict(gridcolor=C["gray200"], linecolor=C["gray200"], zerolinecolor=C["gray200"])

def base_layout(**overrides):
    """Return a layout dict with shared brand styles, merged with overrides."""
    layout = dict(
        paper_bgcolor = C["white"],
        plot_bgcolor  = C["plotBg"],
        font          = dict(family="Inter, Arial, sans-serif", size=12, color=C["gray600"]),
        margin        = dict(t=55, r=70, b=55, l=70),
        legend        = dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=C["gray200"], borderwidth=1),
    )
    layout.update(overrides)
    return layout

PNG_OPTS = dict(format="png", width=1200, height=420, scale=2)

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_HOURS = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]

# ── Run optimizer ─────────────────────────────────────────────────────────────
print("Running BESS arbitrage optimizer (real MILP)...")
result = optimize_bess_arbitrage(
    battery_capacity_mwh = 8.0,
    max_power_mw         = 2.0,
    efficiency           = 0.92,
    optimize_sizing      = False,
    rfnbo_compliant      = False,
)

schedule = result["schedule"]
kpis     = result["kpis"]
cap_mwh  = result["optimal_capacity_mwh"]
T        = len(schedule["Hour"])

# ISO timestamps (2024-01-01T00:00 … 2024-12-31T23:00)
from datetime import datetime, timedelta, timezone
origin     = datetime(2024, 1, 1, tzinfo=timezone.utc)
timestamps = [(origin + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(T)]

prices       = schedule["Price_EUR_MWh"]
charge_mw    = schedule["Charge_MW"]
discharge_mw = schedule["Discharge_MW"]
soc_mwh      = schedule["SoC_MWh"]
revenue_eur  = schedule["Revenue_EUR"]

# Charge shown as negative bars (buying)
charge_neg = [-abs(v) for v in charge_mw]

# Monthly revenue
monthly_rev = []
cursor = 0
for hrs in MONTH_HOURS:
    monthly_rev.append(sum(revenue_eur[cursor:cursor + hrs]))
    cursor += hrs

# Save KPIs as JSON for reference
with open(ROOT / "docs" / "kpis.json", "w") as f:
    json.dump(kpis, f, indent=2)
print(f"KPIs saved to docs/kpis.json")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Dispatch & Market Prices  (1-week window, Jan week 1)
# ─────────────────────────────────────────────────────────────────────────────
WEEK = 24 * 7
ts_w  = timestamps[:WEEK]

fig = go.Figure()

fig.add_trace(go.Bar(
    name   = "Discharge (Selling)",
    x      = ts_w,
    y      = discharge_mw[:WEEK],
    marker = dict(color=C["success"], opacity=0.9),
    yaxis  = "y",
))
fig.add_trace(go.Bar(
    name   = "Charge (Buying)",
    x      = ts_w,
    y      = charge_neg[:WEEK],
    marker = dict(color=C["error"], opacity=0.9),
    yaxis  = "y",
))
fig.add_trace(go.Scatter(
    name = "Price (€/MWh)",
    x    = ts_w,
    y    = prices[:WEEK],
    mode = "lines",
    line = dict(color=C["brand"], dash="dash", width=1.5),
    yaxis = "y2",
))

fig.update_layout(base_layout(
    title   = dict(text="Dispatch & Market Prices — Week 1", font=dict(size=14, color=C["gray900"]), x=0.01),
    barmode = "relative",
    xaxis   = dict(**AXIS_STYLE, type="date", title=dict(text="Date", standoff=8)),
    yaxis   = dict(**AXIS_STYLE, title=dict(text="Power (MW)")),
    yaxis2  = dict(
        gridcolor=C["gray200"], linecolor=C["gray200"], zerolinecolor=C["gray200"],
        title=dict(text="Market Price (€/MWh)"), overlaying="y", side="right", showgrid=False,
    ),
    legend  = dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=C["gray200"], borderwidth=1, xanchor="right", x=0.98, y=1.0),
    height  = 380,
    width   = 1200,
))

path1 = OUT_DIR / "dispatch_snapshot.png"
fig.write_image(str(path1), **PNG_OPTS)
print(f"Saved: {path1}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — State of Energy (SoC)  (1-week window)
# ─────────────────────────────────────────────────────────────────────────────
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    name      = "State of Energy",
    x         = ts_w,
    y         = soc_mwh[:WEEK],
    mode      = "lines",
    line      = dict(color=C["brand"], width=2),
    fill      = "tozeroy",
    fillcolor = C["brandLighter"] + "55",
))
fig2.add_trace(go.Scatter(
    name = "Max Capacity",
    x    = ts_w,
    y    = [cap_mwh] * WEEK,
    mode = "lines",
    line = dict(color=C["brandLight"], dash="dot", width=1.5),
))

fig2.update_layout(base_layout(
    title  = dict(text="State of Energy (SoC) — Week 1", font=dict(size=14, color=C["gray900"]), x=0.01),
    xaxis  = dict(**AXIS_STYLE, type="date", title=dict(text="Date", standoff=8)),
    yaxis  = dict(**AXIS_STYLE, title=dict(text="Energy (MWh)"), rangemode="tozero"),
    legend = dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=C["gray200"], borderwidth=1, xanchor="right", x=0.98, y=1.0),
    height = 360,
    width  = 1200,
))

path2 = OUT_DIR / "soc_snapshot.png"
fig2.write_image(str(path2), **PNG_OPTS)
print(f"Saved: {path2}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Monthly Revenue
# ─────────────────────────────────────────────────────────────────────────────
fig3 = go.Figure()

fig3.add_trace(go.Bar(
    name        = "Monthly Revenue",
    x           = MONTH_NAMES,
    y           = monthly_rev,
    marker      = dict(
        color   = [C["brand"] if v >= 0 else C["error"] for v in monthly_rev],
        opacity = 0.9,
    ),
    text        = [f"€{v:,.0f}" for v in monthly_rev],
    textposition= "outside",
    textfont    = dict(size=11, color=C["gray600"]),
))

fig3.update_layout(base_layout(
    title      = dict(text="Monthly Revenue (€)", font=dict(size=14, color=C["gray900"]), x=0.01),
    xaxis      = dict(**AXIS_STYLE, title=dict(text="Month")),
    yaxis      = dict(**AXIS_STYLE, title=dict(text="Revenue (€)")),
    showlegend = False,
    height     = 360,
    width      = 1200,
))

path3 = OUT_DIR / "monthly_revenue.png"
fig3.write_image(str(path3), **PNG_OPTS)
print(f"Saved: {path3}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — RFNBO comparison  (run optimizer again with RFNBO gate)
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning RFNBO-compliant optimizer run...")
result_rfnbo = optimize_bess_arbitrage(
    battery_capacity_mwh = 8.0,
    max_power_mw         = 2.0,
    efficiency           = 0.92,
    optimize_sizing      = False,
    rfnbo_compliant      = True,
)
rev_rfnbo = result_rfnbo["schedule"]["Revenue_EUR"]

# Cumulative revenue
import numpy as np
cumrev_std   = np.cumsum(revenue_eur)
cumrev_rfnbo = np.cumsum(rev_rfnbo)

monthly_rfnbo = []
cursor = 0
for hrs in MONTH_HOURS:
    monthly_rfnbo.append(sum(rev_rfnbo[cursor:cursor + hrs]))
    cursor += hrs

penalty     = float(cumrev_std[-1] - cumrev_rfnbo[-1])
penalty_pct = penalty / max(abs(cumrev_std[-1]), 1) * 100

# Subplots: cumulative revenue + monthly comparison
fig4 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Cumulative Revenue", "Monthly Comparison"),
    horizontal_spacing=0.1,
)

# Cumulative
fig4.add_trace(go.Scatter(
    name="Unconstrained", x=list(range(T)), y=(cumrev_std / 1000).tolist(),
    mode="lines", line=dict(color=C["brand"], width=2),
), row=1, col=1)

fig4.add_trace(go.Scatter(
    name="RFNBO compliant", x=list(range(T)), y=(cumrev_rfnbo / 1000).tolist(),
    mode="lines", line=dict(color=C["success"], width=2, dash="dash"),
), row=1, col=1)

fig4.add_trace(go.Scatter(
    name="Revenue penalty",
    x=list(range(T)) + list(range(T))[::-1],
    y=(cumrev_std / 1000).tolist() + (cumrev_rfnbo / 1000).tolist()[::-1],
    fill="toself", fillcolor="rgba(239,68,68,0.1)",
    line=dict(color="rgba(0,0,0,0)"),
    showlegend=True,
), row=1, col=1)

# Monthly
x_idx = list(range(len(MONTH_NAMES)))
fig4.add_trace(go.Bar(
    name="Unconstrained", x=MONTH_NAMES,
    y=[v / 1000 for v in monthly_rev],
    marker=dict(color=C["brand"], opacity=0.85),
    offsetgroup=0,
), row=1, col=2)

fig4.add_trace(go.Bar(
    name="RFNBO compliant", x=MONTH_NAMES,
    y=[v / 1000 for v in monthly_rfnbo],
    marker=dict(color=C["success"], opacity=0.85),
    offsetgroup=1,
), row=1, col=2)

fig4.update_layout(
    paper_bgcolor = C["white"],
    plot_bgcolor  = C["plotBg"],
    font          = dict(family="Inter, Arial, sans-serif", size=12, color=C["gray600"]),
    title         = dict(
        text=f"RFNBO Additionality Compliance — Revenue penalty: €{penalty:,.0f} ({penalty_pct:.1f}%)",
        font=dict(size=13, color=C["gray900"]), x=0.01,
    ),
    legend        = dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=C["gray200"], borderwidth=1),
    margin        = dict(t=70, r=40, b=55, l=70),
    barmode       = "group",
    height        = 400,
    width         = 1200,
)
fig4.update_xaxes(gridcolor=C["gray200"], linecolor=C["gray200"], zerolinecolor=C["gray200"])
fig4.update_yaxes(gridcolor=C["gray200"], linecolor=C["gray200"], zerolinecolor=C["gray200"])
fig4.update_xaxes(title_text="Hour of year", row=1, col=1)
fig4.update_yaxes(title_text="Cumulative Revenue (k€)", row=1, col=1)
fig4.update_xaxes(title_text="Month", row=1, col=2)
fig4.update_yaxes(title_text="Revenue (k€)", row=1, col=2)

path4 = OUT_DIR / "rfnbo_analysis.png"
fig4.write_image(str(path4), **PNG_OPTS)
print(f"Saved: {path4}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Week 1: Dispatch vs Price vs SoC
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np

charge_arr = np.array(charge_mw)
disch_arr  = np.array(discharge_mw)
soc_arr    = np.array(soc_mwh)
price_arr  = np.array(prices)
rev_arr    = np.array(revenue_eur)

# Fixed: first 168 hours of the year (Week 1)
hw = list(range(168))
soc_pct = (soc_arr[:168] / cap_mwh * 100).tolist()

fig5 = make_subplots(specs=[[{"secondary_y": True}]])

fig5.add_trace(go.Bar(
    name   = "Charge (buy)",
    x      = hw,
    y      = charge_arr[:168].tolist(),
    marker = dict(color="#2DD4BF", opacity=0.9),
), secondary_y=False)

fig5.add_trace(go.Bar(
    name   = "Discharge (sell)",
    x      = hw,
    y      = (-disch_arr[:168]).tolist(),
    marker = dict(color="#F97316", opacity=0.9),
), secondary_y=False)

fig5.add_trace(go.Scatter(
    name = "Price (€/MWh)",
    x    = hw,
    y    = price_arr[:168].tolist(),
    mode = "lines",
    line = dict(color="#EF4444", width=1.5),
), secondary_y=True)

fig5.add_trace(go.Scatter(
    name = "SoC (%)",
    x    = hw,
    y    = soc_pct,
    mode = "lines",
    line = dict(color="#3B82F6", width=1.5, dash="dash"),
), secondary_y=True)

fig5.update_layout(base_layout(
    title   = dict(
        text="Week 1 — Dispatch vs Price vs SoC",
        font=dict(size=14, color=C["gray900"]), x=0.01,
    ),
    barmode = "relative",
    xaxis   = dict(**AXIS_STYLE, title=dict(text="Hour of week")),
    yaxis   = dict(**AXIS_STYLE, title=dict(text="Power (MW)  [+ charge / − discharge]")),
    yaxis2  = dict(
        **AXIS_STYLE,
        title=dict(text="Grid Price (€/MWh)  /  SoC (%)"),
        overlaying="y", side="right", showgrid=False,
    ),
    legend  = dict(
        bgcolor="rgba(255,255,255,0.9)", bordercolor=C["gray200"], borderwidth=1,
        orientation="h", x=0.5, xanchor="center", y=-0.18,
    ),
    height  = 420,
    width   = 1200,
    margin  = dict(t=55, r=90, b=90, l=70),
))

# Also compute best_week for the print summary at end
weekly_rev = [rev_arr[i*168:(i+1)*168].sum() for i in range(52)]
best_week  = int(np.argmax(weekly_rev))

path5 = OUT_DIR / "peak_week_comparison.png"
fig5.write_image(str(path5), **PNG_OPTS)
print(f"Saved: {path5}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — Forecast vs Perfect Foresight
# Train a GBM price forecaster on the first 8 months, forecast the remaining
# ~4 months, run the MILP on both and compare cumulative revenue.
# ─────────────────────────────────────────────────────────────────────────────
import io as _io
import pandas as _pd
from bess_arbitrage.forecaster import train_price_forecaster, forecast_prices as fc_prices

TRAIN_END  = 5832   # first 8 months (Jan–Aug, 24*243 = 5832)
TEST_START = 5832
TEST_END   = 8760   # Sep–Dec (2928 hours)

print("\nTraining 24h-ahead price forecaster (GBM, scikit-learn)...")
fc_models = train_price_forecaster(price_arr, train_end_h=TRAIN_END)

print("Generating forecasts for test period (Sep–Dec)...")
fc_point, fc_lower, fc_upper = fc_prices(fc_models, price_arr, TEST_START, TEST_END)
actual_test = price_arr[TEST_START:TEST_END]

mae  = float(np.mean(np.abs(fc_point - actual_test)))
rmse = float(np.sqrt(np.mean((fc_point - actual_test) ** 2)))
print(f"[forecaster] MAE={mae:.1f} €/MWh  RMSE={rmse:.1f} €/MWh")

def _arr_to_price_bytes(arr):
    buf = _io.BytesIO()
    _pd.DataFrame({"Price_EUR_MWh": arr}).to_csv(buf, index=False)
    return buf.getvalue()

print("Running test-period MILP — perfect foresight...")
res_perfect = optimize_bess_arbitrage(
    price_bytes          = _arr_to_price_bytes(actual_test),
    battery_capacity_mwh = 8.0,
    max_power_mw         = 2.0,
    efficiency           = 0.92,
    initial_soc          = 0.5,
)
print("Running test-period MILP — forecast-driven...")
res_forecast = optimize_bess_arbitrage(
    price_bytes          = _arr_to_price_bytes(fc_point),
    battery_capacity_mwh = 8.0,
    max_power_mw         = 2.0,
    efficiency           = 0.92,
    initial_soc          = 0.5,
)

# Revenue is evaluated against ACTUAL prices (forecast just drives dispatch)
N_TEST       = TEST_END - TEST_START
actual_arr   = np.array(actual_test)
sqrt_eff_val = np.sqrt(0.92)

fc_charge    = np.array(res_forecast["schedule"]["Charge_MW"])[:N_TEST]
fc_discharge = np.array(res_forecast["schedule"]["Discharge_MW"])[:N_TEST]
rev_forecast_actual = actual_arr * (fc_discharge * sqrt_eff_val - fc_charge / sqrt_eff_val)

rev_perfect  = np.array(res_perfect["schedule"]["Revenue_EUR"])[:N_TEST]

cumrev_perfect  = np.cumsum(rev_perfect)
cumrev_forecast = np.cumsum(rev_forecast_actual)

total_perfect  = float(cumrev_perfect[-1])
total_forecast = float(cumrev_forecast[-1])
capture_rate   = total_forecast / max(abs(total_perfect), 1) * 100

print(f"[forecast] Perfect: €{total_perfect:,.0f}  |  Forecast-driven: €{total_forecast:,.0f}  |  Capture: {capture_rate:.1f}%")

# ── Build figure ──────────────────────────────────────────────────────────────
DISP_H = 14 * 24   # 2-week display window for forecast accuracy panel
hw_fc  = list(range(DISP_H))
hw_all = list(range(TEST_END - TEST_START))

fig6 = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        f"24h-Ahead Price Forecast vs Actual  —  2-Week Window  "
        f"(MAE={mae:.1f} €/MWh, RMSE={rmse:.1f} €/MWh)",
        f"Cumulative Revenue: Perfect Foresight vs Forecast-Driven  "
        f"(Sep–Dec test period, revenue capture {capture_rate:.1f}%)",
    ),
    vertical_spacing=0.18,
)

# ── Top panel: forecast accuracy ──────────────────────────────────────────────
fig6.add_trace(go.Scatter(
    name       = "80% prediction interval",
    x          = hw_fc + hw_fc[::-1],
    y          = fc_upper[:DISP_H].tolist() + fc_lower[:DISP_H].tolist()[::-1],
    fill       = "toself",
    fillcolor  = C["brandLighter"] + "66",
    line       = dict(color="rgba(0,0,0,0)"),
    showlegend = True,
), row=1, col=1)

fig6.add_trace(go.Scatter(
    name = "Forecast (p50)",
    x    = hw_fc,
    y    = fc_point[:DISP_H].tolist(),
    mode = "lines",
    line = dict(color=C["brand"], width=1.5, dash="dash"),
), row=1, col=1)

fig6.add_trace(go.Scatter(
    name = "Actual price",
    x    = hw_fc,
    y    = actual_test[:DISP_H].tolist(),
    mode = "lines",
    line = dict(color=C["error"], width=1.5),
), row=1, col=1)

# ── Bottom panel: cumulative revenue ─────────────────────────────────────────
fig6.add_trace(go.Scatter(
    name = "Perfect foresight",
    x    = hw_all,
    y    = (cumrev_perfect / 1000).tolist(),
    mode = "lines",
    line = dict(color=C["brand"], width=2),
), row=2, col=1)

fig6.add_trace(go.Scatter(
    name = "Forecast-driven",
    x    = hw_all,
    y    = (cumrev_forecast / 1000).tolist(),
    mode = "lines",
    line = dict(color=C["success"], width=2, dash="dash"),
), row=2, col=1)

fig6.add_trace(go.Scatter(
    name      = "Revenue gap",
    x         = hw_all + hw_all[::-1],
    y         = (cumrev_perfect / 1000).tolist() + (cumrev_forecast / 1000).tolist()[::-1],
    fill      = "toself",
    fillcolor = "rgba(239,68,68,0.10)",
    line      = dict(color="rgba(0,0,0,0)"),
    showlegend= True,
), row=2, col=1)

fig6.update_layout(
    paper_bgcolor = C["white"],
    plot_bgcolor  = C["plotBg"],
    font          = dict(family="Inter, Arial, sans-serif", size=12, color=C["gray600"]),
    title         = dict(
        text="Price Forecasting vs Perfect Foresight — Dispatch Revenue Impact",
        font=dict(size=13, color=C["gray900"]), x=0.01,
    ),
    legend = dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=C["gray200"], borderwidth=1),
    margin = dict(t=70, r=60, b=55, l=70),
    height = 720,
    width  = 1200,
)
fig6.update_xaxes(gridcolor=C["gray200"], linecolor=C["gray200"], zerolinecolor=C["gray200"])
fig6.update_yaxes(gridcolor=C["gray200"], linecolor=C["gray200"], zerolinecolor=C["gray200"])
fig6.update_xaxes(title_text="Hour of test period", row=1, col=1)
fig6.update_yaxes(title_text="Price (€/MWh)", row=1, col=1)
fig6.update_xaxes(title_text="Hour of test period", row=2, col=1)
fig6.update_yaxes(title_text="Cumulative Revenue (k€)", row=2, col=1)

path6 = OUT_DIR / "forecast_vs_perfect.png"
fig6.write_image(str(path6), format="png", width=1200, height=720, scale=2)
print(f"Saved: {path6}")

# ─────────────────────────────────────────────────────────────────────────────
# Save combined KPI table as markdown (for README)
# ─────────────────────────────────────────────────────────────────────────────
kpis_rfnbo = result_rfnbo["kpis"]
table_md = f"""| Parameter | Unconstrained | RFNBO Compliant |
|---|---|---|
| Battery Capacity | {kpis['optimal_capacity_mwh']} MWh | {kpis['optimal_capacity_mwh']} MWh |
| Power Rating | {kpis['optimal_power_mw']} MW | {kpis['optimal_power_mw']} MW |
| E:P Ratio | {kpis['ep_ratio_h']} h | {kpis['ep_ratio_h']} h |
| Round-trip Efficiency | {kpis['efficiency_pct']}% | {kpis['efficiency_pct']}% |
| Annual Revenue | €{kpis['annual_revenue_eur']:,.0f} | €{kpis_rfnbo['annual_revenue_eur']:,.0f} |
| Annual Cycles | {kpis['annual_cycles']} | {kpis_rfnbo['annual_cycles']} |
| Total Charged | {kpis['total_charged_mwh']:,.0f} MWh | {kpis_rfnbo['total_charged_mwh']:,.0f} MWh |
| Total Discharged | {kpis['total_discharged_mwh']:,.0f} MWh | {kpis_rfnbo['total_discharged_mwh']:,.0f} MWh |
| Avg Charge Price | {kpis['avg_charge_price_eur_mwh']} €/MWh | {kpis_rfnbo['avg_charge_price_eur_mwh']} €/MWh |
| Avg Discharge Price | {kpis['avg_discharge_price_eur_mwh']} €/MWh | {kpis_rfnbo['avg_discharge_price_eur_mwh']} €/MWh |
| Solver Status | {kpis['solver_status']} | {kpis_rfnbo['solver_status']} |
"""
with open(ROOT / "docs" / "results_table.md", "w") as f:
    f.write(table_md)
print(f"Saved: docs/results_table.md")

print(f"\nAll plots saved to docs/images/")
print(f"Annual revenue (real MILP): €{result['annual_revenue_eur']:,.0f}")
print(f"RFNBO penalty:              €{penalty:,.0f} ({penalty_pct:.1f}%)")
print(f"Peak revenue week:          week {best_week + 1}  (€{weekly_rev[best_week]:,.0f})")