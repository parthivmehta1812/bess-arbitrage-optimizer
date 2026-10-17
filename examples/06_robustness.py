"""
Robustness evaluation — plausibility checks, imputation, and confidence fallback.

Scenarios
---------
S0 : Clean price inputs (no injected faults).
S1 : Faulty inputs — faults injected into test-period hours [1416, 8760),
     single seed=42:
       ~5 % NaN in blocks of 2–24 h
       ~1 % spikes (value × 10)
       ~3 % stuck values in runs of 6–12 h

Strategies (evaluated for both scenarios)
-----------------------------------------
A : GBM, NO robustness checks — naive forward-fill of NaN only, no fallback.
B : GBM + check-and-flag + imputation + day-level confidence/imputation fallback
    (k calibrated on Jan–Feb only; test-period data never touched during
    calibration).
C : Persistence (cleaned price[t-24]) — checks + imputation, no GBM, no fallback.
D : Perfect foresight — MILP on actual prices (benchmark, scenario-independent).

Revenue is always scored on the original, unmodified actual prices.

Output
------
  Printed results table per scenario × strategy, with flag counts for S1.
  docs/images/robustness.png
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
    _milp_schedule,
    _actual_revenue,
    _MONTH_STARTS,
    _MONTH_NAMES,
)
from bess_arbitrage.robustness import (
    _K_CAL_TRAIN_END,
    _K_CAL_END,
    check_and_flag,
    clean_prices,
    naive_ffill,
    choose_k,
    daily_interval_widths,
    select_forecast_per_day,
)

# ── Constants ──────────────────────────────────────────────────────────────────
TEST_START   = _MONTH_STARTS[2]    # 1416 — March, first test month
TEST_END     = _MONTH_STARTS[12]   # 8760 — end of year
N_ESTIMATORS = 250

BESS = dict(
    battery_capacity_mwh=8.0,
    max_power_mw=2.0,
    efficiency=0.92,
    initial_soc=0.5,
)


# ── Fault injection ────────────────────────────────────────────────────────────

def inject_faults(prices: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Inject synthetic faults into test-period hours [TEST_START, TEST_END).

      • ~5 %  NaN — in random blocks of 2–24 consecutive hours
      • ~1 %  spikes — multiply surviving valid hours by 10
      • ~3 %  stuck — freeze value over runs of 6–12 consecutive hours

    Returns a copy; the original array is not modified.
    """
    result = prices.copy()
    rng    = np.random.default_rng(seed)
    n_test = TEST_END - TEST_START        # 7344 h

    # Missing blocks (~5 %)
    target = int(0.05 * n_test)
    filled = 0
    while filled < target:
        blk = int(rng.integers(2, 25))
        off = int(rng.integers(0, max(1, n_test - blk)))
        result[TEST_START + off : TEST_START + off + blk] = np.nan
        filled += blk

    # Spikes on non-NaN hours (~1 %)
    valid  = np.where(~np.isnan(result[TEST_START:TEST_END]))[0]
    n_spk  = max(1, int(0.01 * n_test))
    chosen = rng.choice(valid, size=min(n_spk, len(valid)), replace=False)
    result[chosen + TEST_START] *= 10.0

    # Stuck runs (~3 %)
    target_stuck = int(0.03 * n_test)
    n_stuck      = 0
    for _ in range(3000):
        if n_stuck >= target_stuck:
            break
        run = int(rng.integers(6, 13))
        off = int(rng.integers(0, max(1, n_test - run)))
        a   = TEST_START + off
        if not np.isnan(result[a]):
            result[a : a + run] = result[a]
            n_stuck += run

    return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cap(revenue: float, rev_pf: float) -> float:
    """Revenue capture as percentage of perfect foresight."""
    return revenue / rev_pf * 100.0 if rev_pf > 0 else float("nan")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:  # noqa: PLR0912, PLR0915 (long but deliberate)

    # ── 1. Load prices ────────────────────────────────────────────────────────
    print("Loading 2023 DE price data …")
    prices = load_prices()

    # ── 2. Build and report S1 faulty array ───────────────────────────────────
    print("Injecting faults for scenario S1 (seed=42) …")
    prices_s1_raw = inject_faults(prices, seed=42)

    n_test = TEST_END - TEST_START
    _, inj = check_and_flag(prices_s1_raw[TEST_START:TEST_END])
    print(
        f"  Injected — NaN: {inj['n_nan']} ({inj['n_nan']/n_test*100:.1f}%)  "
        f"Spikes: {inj['n_spike']} ({inj['n_spike']/n_test*100:.1f}%)  "
        f"Stuck: {inj['n_stuck']} ({inj['n_stuck']/n_test*100:.1f}%)"
    )

    # ── 3. Clean both arrays once (for strategies B and C) ────────────────────
    print("Cleaning S0 and S1 full arrays …")
    prices_s0_clean, imp_s0, _   = clean_prices(prices)
    prices_s1_clean, imp_s1, _   = clean_prices(prices_s1_raw)
    prices_s1_ffill              = naive_ffill(prices_s1_raw)   # strategy A / S1

    # ── 4. k calibration on Jan–Feb clean prices only ─────────────────────────
    print("Calibrating fallback threshold k (Jan–Feb only) …")
    k = choose_k(prices)
    print(f"  k = {k:.2f}")

    # ── 5. Seed hist_widths with Feb widths from k-calibration mini-model ─────
    _cal_mdl = train_price_forecaster(prices, train_end_h=_K_CAL_TRAIN_END, n_estimators=100)
    _, p10_feb, p90_feb = forecast_prices(_cal_mdl, prices, _K_CAL_TRAIN_END, _K_CAL_END)
    init_widths = daily_interval_widths(p10_feb, p90_feb)   # 28 Feb days

    hist_s0b: list[float] = list(init_widths)   # grows month by month for S0-B
    hist_s1b: list[float] = list(init_widths)   # grows month by month for S1-B

    # ── 6. Walk-forward monthly loop ──────────────────────────────────────────
    rows: list[dict] = []

    for m_idx in range(2, 12):      # March (0-indexed 2) … December (11)
        mn     = _MONTH_NAMES[m_idx]
        h_s    = _MONTH_STARTS[m_idx]
        h_e    = _MONTH_STARTS[m_idx + 1]
        actual = prices[h_s:h_e]    # true prices — always clean, used to score revenue
        T      = h_e - h_s

        print(f"\n── {mn}  (hours {h_s}–{h_e}, {T} h) {'─'*44}", flush=True)

        # ── D: perfect foresight ──────────────────────────────────────────────
        print(f"  [D ] Perfect foresight MILP …", flush=True)
        c_d, d_d = _milp_schedule(actual, **BESS)
        rev_d    = _actual_revenue(actual, c_d, d_d, BESS["efficiency"])

        # ═══ Scenario S0 (clean) — train once; A and B share GBM ═════════════

        print(f"  [S0] GBM training on hours 168–{h_s} …", flush=True)
        mdl_s0 = train_price_forecaster(
            prices, train_end_h=h_s, n_estimators=N_ESTIMATORS
        )
        p50_s0, p10_s0, p90_s0 = forecast_prices(mdl_s0, prices, h_s, h_e)
        pers_s0 = prices_s0_clean[h_s - 24 : h_e - 24]

        # S0-A: plain GBM, no checks
        print(f"  [S0-A] MILP …", flush=True)
        c_s0a, d_s0a = _milp_schedule(p50_s0, **BESS)
        rev_s0a = _actual_revenue(actual, c_s0a, d_s0a, BESS["efficiency"])
        mae_s0a = float(np.mean(np.abs(actual - p50_s0)))

        # S0-B: GBM + confidence / imputation fallback
        w_s0 = daily_interval_widths(p10_s0, p90_s0)
        sel_s0b, br_s0b = select_forecast_per_day(
            p50_s0, p10_s0, p90_s0, pers_s0,
            imp_s0, np.array(hist_s0b), k, h_offset=h_s,
        )
        print(f"  [S0-B] MILP …", flush=True)
        c_s0b, d_s0b = _milp_schedule(sel_s0b, **BESS)
        rev_s0b = _actual_revenue(actual, c_s0b, d_s0b, BESS["efficiency"])
        mae_s0b = float(np.mean(np.abs(actual - sel_s0b)))
        fb_s0b  = br_s0b.count("persistence") / len(br_s0b) * 100.0

        # S0-C: cleaned persistence only
        print(f"  [S0-C] MILP …", flush=True)
        c_s0c, d_s0c = _milp_schedule(pers_s0, **BESS)
        rev_s0c = _actual_revenue(actual, c_s0c, d_s0c, BESS["efficiency"])
        mae_s0c = float(np.mean(np.abs(actual - pers_s0)))

        # ═══ Scenario S1 (faulty) ════════════════════════════════════════════

        # S1-A: GBM trained on forward-filled faulty prices (no checks)
        print(f"  [S1-A] GBM training (ffill) …", flush=True)
        mdl_s1a = train_price_forecaster(
            prices_s1_ffill, train_end_h=h_s, n_estimators=N_ESTIMATORS
        )
        p50_s1a, _, _ = forecast_prices(mdl_s1a, prices_s1_ffill, h_s, h_e)
        print(f"  [S1-A] MILP …", flush=True)
        c_s1a, d_s1a = _milp_schedule(p50_s1a, **BESS)
        rev_s1a = _actual_revenue(actual, c_s1a, d_s1a, BESS["efficiency"])
        mae_s1a = float(np.mean(np.abs(actual - p50_s1a)))

        # S1-B: GBM on cleaned faulty prices + fallback
        print(f"  [S1-B] GBM training (cleaned) …", flush=True)
        mdl_s1b = train_price_forecaster(
            prices_s1_clean, train_end_h=h_s, n_estimators=N_ESTIMATORS
        )
        p50_s1b, p10_s1b, p90_s1b = forecast_prices(
            mdl_s1b, prices_s1_clean, h_s, h_e
        )
        pers_s1 = prices_s1_clean[h_s - 24 : h_e - 24]

        w_s1 = daily_interval_widths(p10_s1b, p90_s1b)
        sel_s1b, br_s1b = select_forecast_per_day(
            p50_s1b, p10_s1b, p90_s1b, pers_s1,
            imp_s1, np.array(hist_s1b), k, h_offset=h_s,
        )
        print(f"  [S1-B] MILP …", flush=True)
        c_s1b, d_s1b = _milp_schedule(sel_s1b, **BESS)
        rev_s1b = _actual_revenue(actual, c_s1b, d_s1b, BESS["efficiency"])
        mae_s1b = float(np.mean(np.abs(actual - sel_s1b)))
        fb_s1b  = br_s1b.count("persistence") / len(br_s1b) * 100.0

        # S1-C: cleaned persistence
        print(f"  [S1-C] MILP …", flush=True)
        c_s1c, d_s1c = _milp_schedule(pers_s1, **BESS)
        rev_s1c = _actual_revenue(actual, c_s1c, d_s1c, BESS["efficiency"])
        mae_s1c = float(np.mean(np.abs(actual - pers_s1)))

        # ── Flag counts for this month's slice (approximation) ─────────────────
        _, fl   = check_and_flag(prices_s1_raw[h_s:h_e])
        n_imp_m = int(imp_s1[h_s:h_e].sum())

        # ── Update hist_widths for next month ──────────────────────────────────
        hist_s0b.extend(w_s0.tolist())
        hist_s1b.extend(w_s1.tolist())

        rows.append({
            "month":    mn,
            "n_h":      T,
            "rev_D":    rev_d,
            "rev_s0a":  rev_s0a, "mae_s0a": mae_s0a, "fb_s0a": float("nan"),
            "rev_s0b":  rev_s0b, "mae_s0b": mae_s0b, "fb_s0b": fb_s0b,
            "rev_s0c":  rev_s0c, "mae_s0c": mae_s0c, "fb_s0c": float("nan"),
            "rev_s1a":  rev_s1a, "mae_s1a": mae_s1a, "fb_s1a": float("nan"),
            "rev_s1b":  rev_s1b, "mae_s1b": mae_s1b, "fb_s1b": fb_s1b,
            "rev_s1c":  rev_s1c, "mae_s1c": mae_s1c, "fb_s1c": float("nan"),
            "n_nan":    fl["n_nan"],
            "n_spk":    fl["n_spike"],
            "n_stk":    fl["n_stuck"],
            "n_imp":    n_imp_m,
        })

        print(
            f"  D=€{rev_d:,.0f}  "
            f"S0: A={_cap(rev_s0a,rev_d):.1f}% B={_cap(rev_s0b,rev_d):.1f}%"
            f"(fb={fb_s0b:.0f}%) C={_cap(rev_s0c,rev_d):.1f}%  "
            f"S1: A={_cap(rev_s1a,rev_d):.1f}% B={_cap(rev_s1b,rev_d):.1f}%"
            f"(fb={fb_s1b:.0f}%) C={_cap(rev_s1c,rev_d):.1f}%",
            flush=True,
        )

    # ── Print summary tables ───────────────────────────────────────────────────
    _print_tables(rows, k)

    # ── Plot ───────────────────────────────────────────────────────────────────
    _plot(rows)


# ── Output helpers ─────────────────────────────────────────────────────────────

def _overall(rows: list[dict]) -> dict:
    """Aggregate rows into one overall summary."""
    total_d   = sum(r["rev_D"]   for r in rows)
    total_s0a = sum(r["rev_s0a"] for r in rows)
    total_s0b = sum(r["rev_s0b"] for r in rows)
    total_s0c = sum(r["rev_s0c"] for r in rows)
    total_s1a = sum(r["rev_s1a"] for r in rows)
    total_s1b = sum(r["rev_s1b"] for r in rows)
    total_s1c = sum(r["rev_s1c"] for r in rows)
    w  = np.array([r["n_h"] for r in rows], dtype=float)
    wm = lambda key: float(np.average([r[key] for r in rows], weights=w))
    return {
        "month": "Overall",
        "n_h": int(w.sum()),
        "rev_D":   total_d,
        "rev_s0a": total_s0a, "mae_s0a": wm("mae_s0a"), "fb_s0a": float("nan"),
        "rev_s0b": total_s0b, "mae_s0b": wm("mae_s0b"), "fb_s0b": wm("fb_s0b"),
        "rev_s0c": total_s0c, "mae_s0c": wm("mae_s0c"), "fb_s0c": float("nan"),
        "rev_s1a": total_s1a, "mae_s1a": wm("mae_s1a"), "fb_s1a": float("nan"),
        "rev_s1b": total_s1b, "mae_s1b": wm("mae_s1b"), "fb_s1b": wm("fb_s1b"),
        "rev_s1c": total_s1c, "mae_s1c": wm("mae_s1c"), "fb_s1c": float("nan"),
        "n_nan": sum(r["n_nan"] for r in rows),
        "n_spk": sum(r["n_spk"] for r in rows),
        "n_stk": sum(r["n_stk"] for r in rows),
        "n_imp": sum(r["n_imp"] for r in rows),
    }


def _print_tables(rows: list[dict], k: float) -> None:
    ov  = _overall(rows)
    all_rows = rows + [ov]

    print()
    print("=" * 95)
    print(f"  Robustness evaluation — k = {k:.2f}  |  BESS 8 MWh / 2 MW / η=0.92")
    print("=" * 95)

    # ── Scenario S0 ───────────────────────────────────────────────────────────
    print()
    print("  Scenario S0 — clean inputs")
    hdr = (
        f"{'Month':<8} {'N-h':>5}  {'PF rev':>9}  "
        f"{'A cap%':>7} {'A MAE':>7}  "
        f"{'B cap%':>7} {'B MAE':>7} {'B fb%':>6}  "
        f"{'C cap%':>7} {'C MAE':>7}"
    )
    print("  " + hdr)
    print("  " + "-" * len(hdr))
    for r in all_rows:
        sep = "=" if r["month"] == "Overall" else " "
        if r["month"] == "Overall":
            print("  " + "-" * len(hdr))
        d = r["rev_D"]
        print(
            f"  {r['month']:<8} {r['n_h']:>5}  "
            f"€{d:>8,.0f}  "
            f"{_cap(r['rev_s0a'],d):>7.1f} {r['mae_s0a']:>7.1f}  "
            f"{_cap(r['rev_s0b'],d):>7.1f} {r['mae_s0b']:>7.1f} {r['fb_s0b']:>6.1f}  "
            f"{_cap(r['rev_s0c'],d):>7.1f} {r['mae_s0c']:>7.1f}"
        )
    print("  " + "=" * len(hdr))

    # ── Scenario S1 ───────────────────────────────────────────────────────────
    print()
    print("  Scenario S1 — faulty inputs (seed=42)")
    hdr2 = (
        f"{'Month':<8} {'N-h':>5}  "
        f"{'NaN':>4} {'Spk':>4} {'Stk':>4} {'Imp':>4}  "
        f"{'A cap%':>7} {'A MAE':>7}  "
        f"{'B cap%':>7} {'B MAE':>7} {'B fb%':>6}  "
        f"{'C cap%':>7} {'C MAE':>7}"
    )
    print("  " + hdr2)
    print("  " + "-" * len(hdr2))
    for r in all_rows:
        if r["month"] == "Overall":
            print("  " + "-" * len(hdr2))
        d = r["rev_D"]
        print(
            f"  {r['month']:<8} {r['n_h']:>5}  "
            f"{r['n_nan']:>4} {r['n_spk']:>4} {r['n_stk']:>4} {r['n_imp']:>4}  "
            f"{_cap(r['rev_s1a'],d):>7.1f} {r['mae_s1a']:>7.1f}  "
            f"{_cap(r['rev_s1b'],d):>7.1f} {r['mae_s1b']:>7.1f} {r['fb_s1b']:>6.1f}  "
            f"{_cap(r['rev_s1c'],d):>7.1f} {r['mae_s1c']:>7.1f}"
        )
    print("  " + "=" * len(hdr2))
    print()


def _plot(rows: list[dict]) -> None:
    OUT_DIR = ROOT / "docs" / "images"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        print(f"\n[plot] Skipped (missing dependency: {exc})")
        return

    C = {
        "brand":      "#4C09E7",
        "brand2":     "#A485F6",
        "orange":     "#F97316",
        "orange2":    "#FCD34D",
        "teal":       "#14B8A6",
        "teal2":      "#99F6E4",
        "success":    "#22C55E",
        "gray900":    "#18122B",
        "gray600":    "#574F7D",
        "gray200":    "#DCD7EC",
        "white":      "#ffffff",
        "plotBg":     "#faf9fe",
    }
    AXIS = dict(gridcolor=C["gray200"], linecolor=C["gray200"], zerolinecolor=C["gray200"])

    months  = [r["month"] for r in rows]
    x       = list(range(len(months)))
    bar_w   = 0.25
    x_gap   = 0.28   # half-spacing between A/B/C within a subplot

    def caps(key: str) -> list[float]:
        return [_cap(r[key], r["rev_D"]) for r in rows]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("S0 — clean inputs", "S1 — faulty inputs (seed=42)"),
        horizontal_spacing=0.12,
    )

    def add_bars(col: int, suffix: str, colors: list[str]) -> None:
        labels = ["A (GBM)", "B (GBM+fallback)", "C (persistence)"]
        keys   = [f"rev_s{suffix}a", f"rev_s{suffix}b", f"rev_s{suffix}c"]
        offsets = [-x_gap, 0, x_gap]
        for (lbl, key, off, clr) in zip(labels, keys, offsets, colors):
            ys = caps(key)
            fig.add_trace(go.Bar(
                name=lbl,
                x=[xi + off for xi in x],
                y=ys,
                width=bar_w,
                marker=dict(color=clr, opacity=0.85),
                text=[f"{v:.0f}%" for v in ys],
                textposition="outside",
                textfont=dict(size=8, color=C["gray600"]),
                showlegend=(col == 1),
            ), row=1, col=col)
        fig.add_trace(go.Scatter(
            name="Perfect foresight (100%)",
            x=[-0.7, len(months) - 0.3],
            y=[100, 100],
            mode="lines",
            line=dict(color=C["success"], dash="dash", width=1.5),
            showlegend=(col == 1),
        ), row=1, col=col)

    add_bars(1, "0", [C["brand"], C["brand2"], C["teal"]])
    add_bars(2, "1", [C["orange"], C["orange2"], C["teal2"]])

    ov = _overall(rows)
    fig.update_layout(
        paper_bgcolor=C["white"],
        plot_bgcolor=C["plotBg"],
        font=dict(family="Inter, Arial, sans-serif", size=11, color=C["gray600"]),
        title=dict(
            text=(
                "Robustness Evaluation (Mar–Dec 2023)  —  "
                f"S0 overall: A={_cap(ov['rev_s0a'],ov['rev_D']):.1f}%  "
                f"B={_cap(ov['rev_s0b'],ov['rev_D']):.1f}%  "
                f"C={_cap(ov['rev_s0c'],ov['rev_D']):.1f}%  |  "
                f"S1: A={_cap(ov['rev_s1a'],ov['rev_D']):.1f}%  "
                f"B={_cap(ov['rev_s1b'],ov['rev_D']):.1f}%  "
                f"C={_cap(ov['rev_s1c'],ov['rev_D']):.1f}%"
            ),
            font=dict(size=11, color=C["gray900"]), x=0.01,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)", bordercolor=C["gray200"], borderwidth=1
        ),
        margin=dict(t=80, r=40, b=70, l=70),
        height=460, width=1300,
        barmode="group",
    )
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    for col in (1, 2):
        fig.update_xaxes(tickvals=x, ticktext=months, tickangle=-30, row=1, col=col)
        fig.update_yaxes(title_text="Revenue capture (%)", row=1, col=col, range=[0, 115])

    path = OUT_DIR / "robustness.png"
    fig.write_image(str(path), format="png", width=1300, height=460, scale=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
