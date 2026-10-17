"""
Forecast recalibration via Conformal Quantile Regression (CQR).

Data split (hourly 2023 DE prices, 8 760 h, no look-ahead anywhere)
--------------------------------------------------------------------
  Train       : hours  168 –  5 088  (Jan–Jul, ~4 920 h)
  Calibration : hours 5 088 –  5 832  (August,    744 h)
  Test        : hours 5 832 –  8 760  (Sep–Dec,  2 928 h)

Three variants evaluated on the test period
-------------------------------------------
  (a) Uncalibrated  — GBM quantiles straight from training
  (b) Static CQR    — single offset learned on August
  (c) Rolling CQR   — 30-day adaptive window, no look-ahead

Metrics per variant
-------------------
  p10 coverage (target 10%), p90 coverage (target 90%),
  80% interval coverage (target 80%), mean interval width (EUR/MWh),
  pinball loss p10 / p50 / p90, CRPS approximation.

Also writes docs/images/forecast_recalibration.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bess_arbitrage.loader import load_prices
from bess_arbitrage.forecaster import train_price_forecaster, forecast_prices
from bess_arbitrage.calibration import conformal_quantile_offset, rolling_conformal

# ── Data split ────────────────────────────────────────────────────────────────
TRAIN_START = 168      # need 168 h history for lag features
TRAIN_END   = 5_088    # Jan–Jul
CAL_START   = 5_088    # August
CAL_END     = 5_832
TEST_START  = 5_832    # Sep–Dec
TEST_END    = 8_760

# ── Load prices ───────────────────────────────────────────────────────────────
print("Loading 2023 DE price data...")
prices = load_prices()                    # bundled dataset, shape (8760,)

actual_cal  = prices[CAL_START:CAL_END]
actual_test = prices[TEST_START:TEST_END]

# ── Train GBM forecaster on Jan–Jul only ─────────────────────────────────────
print(f"Training forecaster on hours {TRAIN_START}–{TRAIN_END} (Jan–Jul)...")
models = train_price_forecaster(prices, train_end_h=TRAIN_END)

# ── Forecast calibration period (August) ─────────────────────────────────────
print("Forecasting calibration period (August)...")
q50_cal, q10_cal, q90_cal = forecast_prices(models, prices, CAL_START, CAL_END)

# ── Forecast test period (Sep–Dec) ───────────────────────────────────────────
print("Forecasting test period (Sep–Dec)...")
q50_test, q10_test, q90_test = forecast_prices(models, prices, TEST_START, TEST_END)

# ── (b) Static CQR — offsets learned on August ───────────────────────────────
off10_static = conformal_quantile_offset(actual_cal, q10_cal, 0.10)
off90_static = conformal_quantile_offset(actual_cal, q90_cal, 0.90)

q10_static = q10_test + off10_static
q90_static = q90_test + off90_static
# Enforce p10 <= p50 <= p90
stacked = np.sort(np.stack([q10_static, q50_test, q90_static], axis=1), axis=1)
q10_static, q50_static, q90_static = stacked[:, 0], stacked[:, 1], stacked[:, 2]

print(f"[static CQR]  off10={off10_static:+.2f}  off90={off90_static:+.2f} EUR/MWh")

# ── (c) Rolling CQR — 30-day adaptive window ─────────────────────────────────
print("Computing rolling CQR (30-day window)...")
q10_roll, q50_roll, q90_roll = rolling_conformal(
    actual_cal, q10_cal, q90_cal,
    actual_test, q10_test, q50_test, q90_test,
    window_days=30,
)

# ── Metrics ───────────────────────────────────────────────────────────────────

def pinball(alpha: float, y: np.ndarray, q: np.ndarray) -> float:
    return float(np.where(y >= q, alpha * (y - q), (1 - alpha) * (q - y)).mean())


def metrics(y: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> dict:
    pb10 = pinball(0.10, y, q10)
    pb50 = pinball(0.50, y, q50)
    pb90 = pinball(0.90, y, q90)
    return {
        "p10_cov":       float(np.mean(y <= q10)) * 100,
        "p90_cov":       float(np.mean(y <= q90)) * 100,
        "int80_cov":     float(np.mean((y >= q10) & (y <= q90))) * 100,
        "width":         float(np.mean(q90 - q10)),
        "pb10":          pb10,
        "pb50":          pb50,
        "pb90":          pb90,
        "crps":          2 * np.mean([pb10, pb50, pb90]),
    }


m_raw    = metrics(actual_test, q10_test,   q50_test,   q90_test)
m_static = metrics(actual_test, q10_static, q50_static, q90_static)
m_roll   = metrics(actual_test, q10_roll,   q50_roll,   q90_roll)

# ── Results table ─────────────────────────────────────────────────────────────
VARIANTS = [
    ("(a) Uncalibrated",  m_raw),
    ("(b) Static CQR",    m_static),
    ("(c) Rolling CQR",   m_roll),
]

COL_W = 20

header = (
    f"{'Metric':<28}"
    + "".join(f"{v[0]:>{COL_W}}" for v in VARIANTS)
)
sep = "-" * len(header)

rows = [
    ("p10 coverage %  (target 10%)", "p10_cov",   "{:.1f}%"),
    ("p90 coverage %  (target 90%)", "p90_cov",   "{:.1f}%"),
    ("80% intv coverage (target 80%)", "int80_cov", "{:.1f}%"),
    ("Mean interval width (EUR/MWh)", "width",     "{:.2f}"),
    ("Pinball loss p10  (EUR/MWh)",  "pb10",      "{:.3f}"),
    ("Pinball loss p50  (EUR/MWh)",  "pb50",      "{:.3f}"),
    ("Pinball loss p90  (EUR/MWh)",  "pb90",      "{:.3f}"),
    ("CRPS             (EUR/MWh)",   "crps",      "{:.3f}"),
]

print()
print("=" * len(header))
print("  Forecast recalibration results — test period Sep–Dec 2023")
print("=" * len(header))
print(header)
print(sep)
for label, key, fmt in rows:
    line = f"{label:<28}"
    for _, m in VARIANTS:
        val = fmt.format(m[key])
        line += f"{val:>{COL_W}}"
    print(line)
print(sep)
print()

# ── Reliability diagram ───────────────────────────────────────────────────────
OUT_DIR = ROOT / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.interpolate import interp1d

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
    AXIS_STYLE = dict(gridcolor=C["gray200"], linecolor=C["gray200"], zerolinecolor=C["gray200"])

    # Build fine-grained reliability curves for each variant
    q_vals = np.array([0.10, 0.50, 0.90])
    alphas_fine = np.linspace(0.05, 0.95, 19)

    def reliability_curve(y, q10, q50, q90):
        q_preds = np.stack([q10, q50, q90], axis=1)
        emp = []
        for alpha in alphas_fine:
            q_a = np.array([
                float(interp1d(q_vals, q_preds[i], fill_value="extrapolate")(alpha))
                for i in range(len(q_preds))
            ])
            emp.append(float(np.mean(y[:len(q_a)] <= q_a)))
        return emp

    emp_raw    = reliability_curve(actual_test, q10_test,   q50_test,   q90_test)
    emp_static = reliability_curve(actual_test, q10_static, q50_static, q90_static)
    emp_roll   = reliability_curve(actual_test, q10_roll,   q50_roll,   q90_roll)

    diag = alphas_fine.tolist()

    # Pinball bars for each variant
    alphas3 = [0.10, 0.50, 0.90]
    labels3 = ["p10", "p50", "p90"]

    pb_raw    = [pinball(a, actual_test, q) for a, q in zip(alphas3, [q10_test,   q50_test,   q90_test])]
    pb_static = [pinball(a, actual_test, q) for a, q in zip(alphas3, [q10_static, q50_static, q90_static])]
    pb_roll   = [pinball(a, actual_test, q) for a, q in zip(alphas3, [q10_roll,   q50_roll,   q90_roll])]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Reliability diagram", "Pinball loss by quantile"),
        horizontal_spacing=0.12,
    )

    # Perfect calibration diagonal
    fig.add_trace(go.Scatter(
        name="Perfect calibration", x=diag, y=diag,
        mode="lines", line=dict(color=C["gray600"], dash="dash", width=1.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        name="(a) Uncalibrated",
        x=alphas_fine.tolist(), y=emp_raw,
        mode="lines+markers",
        line=dict(color=C["error"], width=2),
        marker=dict(size=5, color=C["error"]),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        name="(b) Static CQR",
        x=alphas_fine.tolist(), y=emp_static,
        mode="lines+markers",
        line=dict(color=C["brand"], width=2),
        marker=dict(size=5, color=C["brand"]),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        name="(c) Rolling CQR",
        x=alphas_fine.tolist(), y=emp_roll,
        mode="lines+markers",
        line=dict(color=C["success"], width=2, dash="dot"),
        marker=dict(size=5, color=C["success"]),
    ), row=1, col=1)

    # Pinball bars — grouped
    x_pos = list(range(len(labels3)))
    bar_w = 0.25

    for i, (name, pbs, color) in enumerate([
        ("(a) Uncalibrated", pb_raw,    C["error"]),
        ("(b) Static CQR",   pb_static, C["brand"]),
        ("(c) Rolling CQR",  pb_roll,   C["success"]),
    ]):
        fig.add_trace(go.Bar(
            name=name,
            x=[l + (i - 1) * bar_w for l in x_pos],
            y=pbs,
            width=bar_w,
            marker=dict(color=color, opacity=0.85),
            text=[f"{v:.2f}" for v in pbs],
            textposition="outside",
            textfont=dict(size=10, color=C["gray600"]),
            showlegend=False,
        ), row=1, col=2)

    fig.update_layout(
        paper_bgcolor=C["white"],
        plot_bgcolor=C["plotBg"],
        font=dict(family="Inter, Arial, sans-serif", size=12, color=C["gray600"]),
        title=dict(
            text=(
                f"Forecast Recalibration (CQR) — "
                f"p90 coverage: "
                f"(a) {m_raw['p90_cov']:.1f}%  "
                f"(b) {m_static['p90_cov']:.1f}%  "
                f"(c) {m_roll['p90_cov']:.1f}%  "
                f"(target 90%)"
            ),
            font=dict(size=12, color=C["gray900"]), x=0.01,
        ),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=C["gray200"], borderwidth=1),
        margin=dict(t=70, r=40, b=55, l=70),
        height=440, width=1200,
    )
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    fig.update_xaxes(title_text="Predicted quantile", row=1, col=1)
    fig.update_yaxes(title_text="Empirical coverage", row=1, col=1)
    fig.update_xaxes(
        title_text="Quantile", row=1, col=2,
        tickvals=x_pos, ticktext=labels3,
    )
    fig.update_yaxes(title_text="Pinball loss (EUR/MWh)", row=1, col=2)

    plot_path = OUT_DIR / "forecast_recalibration.png"
    fig.write_image(str(plot_path), format="png", width=1200, height=440, scale=2)
    print(f"Saved: {plot_path}")

except ImportError as exc:
    print(f"[plot] Skipped (missing dependency: {exc})")
