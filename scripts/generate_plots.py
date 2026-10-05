"""
Generate result plots for the README.
Style matches the LCOxEngine dashboard: light card theme, purple/green/teal palette.

Run once:
    pip install matplotlib pandas numpy
    python scripts/generate_plots.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Brand palette (matches LCOxEngine frontend) ───────────────────────────────
PURPLE  = "#4C09E7"
DPURPLE = "#2D156E"
GREEN   = "#22C55E"
RED     = "#EF4444"
TEAL    = "#2DD4BF"
BLUE    = "#3B82F6"
LGRAY   = "#F5F1FE"
BORDER  = "#DCD7EC"
MGRAY   = "#6E6599"
DARK    = "#18122B"

# ── Global style ──────────────────────────────────────────────────────────────
rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          9,
    "axes.facecolor":     "white",
    "figure.facecolor":   LGRAY,
    "axes.edgecolor":     BORDER,
    "axes.linewidth":     1.0,
    "axes.grid":          True,
    "grid.color":         BORDER,
    "grid.linewidth":     0.6,
    "grid.alpha":         1.0,
    "xtick.color":        MGRAY,
    "ytick.color":        MGRAY,
    "axes.labelcolor":    DARK,
    "text.color":         DARK,
    "legend.frameon":     True,
    "legend.framealpha":  1.0,
    "legend.edgecolor":   BORDER,
    "legend.facecolor":   "white",
    "figure.dpi":         150,
})

def card_fig(nrows, ncols, figsize, title):
    """Create a figure styled like a dashboard card."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(LGRAY)
    fig.suptitle(title, fontsize=12, fontweight="bold", color=DARK,
                 x=0.5, y=0.98, va="top")
    # Purple top accent bar
    fig.patches.append(
        mpatches.FancyBboxPatch(
            (0.02, 0.96), 0.96, 0.008,
            boxstyle="square,pad=0",
            transform=fig.transFigure,
            color=PURPLE, zorder=10, clip_on=False,
        )
    )
    return fig, axes

# ── Synthetic price + schedule ────────────────────────────────────────────────
rng = np.random.default_rng(42)
hours = np.arange(8760)
hour_of_day = hours % 24

base_price = 35 + 55 * np.clip(np.sin(np.pi * (hour_of_day - 6) / 14), 0, 1)
seasonal   = 10 * np.cos(2 * np.pi * hours / 8760)
noise      = rng.normal(0, 8, 8760)
prices     = np.clip(base_price + seasonal + noise, -10, 200)

capacity   = 8.0
max_power  = 2.0
efficiency = 0.92
sqrt_eff   = math.sqrt(efficiency)

soc      = np.zeros(8760)
charge   = np.zeros(8760)
discharge = np.zeros(8760)
soc[0]   = capacity * 0.5

for t in range(1, 8760):
    prev = soc[t - 1]
    if prices[t] < 30 and prev < capacity - 0.1:
        c = min(max_power, (capacity - prev) / sqrt_eff)
        charge[t] = c
        soc[t] = prev + c * sqrt_eff
    elif prices[t] > 70 and prev > 0.5:
        d = min(max_power, prev * sqrt_eff)
        discharge[t] = d
        soc[t] = prev - d / sqrt_eff
    else:
        soc[t] = prev

revenue_per_hour = prices * (discharge * sqrt_eff - charge / sqrt_eff)
cumulative_rev   = np.cumsum(revenue_per_hour)

month_hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — 2-week dispatch snapshot
# ─────────────────────────────────────────────────────────────────────────────
WINDOW = 24 * 14
t_w = np.arange(WINDOW)

fig, axes = card_fig(3, 1, (13, 8), "BESS Arbitrage — 2-Week Dispatch Snapshot")
fig.subplots_adjust(hspace=0.45, left=0.08, right=0.97, top=0.93, bottom=0.07)

# Price
ax = axes[0]
ax.plot(t_w, prices[:WINDOW], color=PURPLE, linewidth=1.2, label="Day-ahead price")
ax.axhline(30, color=TEAL, linewidth=1.0, linestyle="--", label="Charge threshold (30 €/MWh)")
ax.axhline(70, color=RED,  linewidth=1.0, linestyle="--", label="Discharge threshold (70 €/MWh)")
ax.fill_between(t_w, 30, prices[:WINDOW],
                where=prices[:WINDOW] < 30, alpha=0.12, color=TEAL)
ax.fill_between(t_w, 70, prices[:WINDOW],
                where=prices[:WINDOW] > 70, alpha=0.12, color=RED)
ax.set_ylabel("Price (EUR/MWh)", color=MGRAY, fontsize=8)
ax.set_title("Day-Ahead Electricity Price", fontsize=9, color=DPURPLE, fontweight="bold", loc="left")
ax.legend(fontsize=7.5, loc="upper right")

# Charge / discharge
ax = axes[1]
ax.bar(t_w,  charge[:WINDOW],    color=TEAL, alpha=0.85, label="Charge (MW)",    width=1)
ax.bar(t_w, -discharge[:WINDOW], color=PURPLE, alpha=0.85, label="Discharge (MW)", width=1)
ax.axhline(0, color=BORDER, linewidth=0.8)
ax.set_ylabel("Power (MW)", color=MGRAY, fontsize=8)
ax.set_title("Charge / Discharge Schedule", fontsize=9, color=DPURPLE, fontweight="bold", loc="left")
ax.legend(fontsize=7.5, loc="upper right")

# SoC
ax = axes[2]
ax.fill_between(t_w, soc[:WINDOW], alpha=0.18, color=GREEN)
ax.plot(t_w, soc[:WINDOW], color=GREEN, linewidth=1.5, label="State of Charge (MWh)")
ax.axhline(capacity, color=MGRAY, linewidth=0.8, linestyle=":", label=f"Capacity ({capacity} MWh)")
ax.set_ylabel("SoC (MWh)", color=MGRAY, fontsize=8)
ax.set_xlabel("Hour of year", color=MGRAY, fontsize=8)
ax.set_title("State of Charge", fontsize=9, color=DPURPLE, fontweight="bold", loc="left")
ax.legend(fontsize=7.5, loc="upper right")

path1 = OUT_DIR / "dispatch_snapshot.png"
fig.savefig(path1, bbox_inches="tight", facecolor=LGRAY)
plt.close()
print(f"Saved: {path1}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Annual summary
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = card_fig(1, 2, (13, 4.5), "Annual Performance Summary")
fig.subplots_adjust(wspace=0.35, left=0.08, right=0.97, top=0.88, bottom=0.14)

# Cumulative revenue
ax = axes[0]
ax.plot(hours, cumulative_rev / 1000, color=PURPLE, linewidth=1.5)
ax.fill_between(hours, cumulative_rev / 1000, alpha=0.12, color=PURPLE)
ax.set_xlabel("Hour of year", color=MGRAY, fontsize=8)
ax.set_ylabel("Cumulative Revenue (k€)", color=MGRAY, fontsize=8)
ax.set_title(f"Total Annual Revenue: €{cumulative_rev[-1]:,.0f}",
             fontsize=9, color=DPURPLE, fontweight="bold", loc="left")

# Price distribution
ax = axes[1]
ax.hist(prices, bins=80, color=MGRAY, alpha=0.4, label="All hours")
ax.hist(prices[charge > 0.01],    bins=40, color=TEAL,   alpha=0.8, label="Charge hours")
ax.hist(prices[discharge > 0.01], bins=40, color=PURPLE, alpha=0.8, label="Discharge hours")
ax.set_xlabel("Price (EUR/MWh)", color=MGRAY, fontsize=8)
ax.set_ylabel("Hours per year", color=MGRAY, fontsize=8)
ax.set_title("Price Distribution by Operation Mode",
             fontsize=9, color=DPURPLE, fontweight="bold", loc="left")
ax.legend(fontsize=7.5)

path2 = OUT_DIR / "annual_summary.png"
fig.savefig(path2, bbox_inches="tight", facecolor=LGRAY)
plt.close()
print(f"Saved: {path2}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Monthly revenue
# ─────────────────────────────────────────────────────────────────────────────
monthly_rev = []
start = 0
for h in month_hours:
    monthly_rev.append(revenue_per_hour[start:start+h].sum())
    start += h

fig, ax = plt.subplots(figsize=(12, 4.5))
fig.patch.set_facecolor(LGRAY)
ax.set_facecolor("white")
fig.patches.append(
    mpatches.FancyBboxPatch(
        (0.02, 0.96), 0.96, 0.008,
        boxstyle="square,pad=0",
        transform=fig.transFigure,
        color=PURPLE, zorder=10, clip_on=False,
    )
)

colors = [GREEN if r >= 0 else RED for r in monthly_rev]
bars = ax.bar(month_names, [r/1000 for r in monthly_rev],
              color=colors, alpha=0.88, edgecolor="white", linewidth=0.5)
ax.axhline(0, color=BORDER, linewidth=0.8)
ax.set_ylabel("Revenue (k€)", color=MGRAY, fontsize=8)
ax.set_title("Monthly Arbitrage Revenue", fontsize=12, fontweight="bold",
             color=DARK, pad=12)
ax.tick_params(colors=MGRAY)
for spine in ax.spines.values():
    spine.set_edgecolor(BORDER)

for bar, val in zip(bars, monthly_rev):
    ypos = bar.get_height() + (0.015 if val >= 0 else -0.08)
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f"€{val:,.0f}", ha="center", va="bottom", fontsize=7.5, color=DARK)

plt.tight_layout()
path3 = OUT_DIR / "monthly_revenue.png"
fig.savefig(path3, bbox_inches="tight", facecolor=LGRAY)
plt.close()
print(f"Saved: {path3}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — RFNBO compliance
# ─────────────────────────────────────────────────────────────────────────────
soc_rfnbo       = np.zeros(8760)
charge_rfnbo    = np.zeros(8760)
discharge_rfnbo = np.zeros(8760)
soc_rfnbo[0]   = capacity * 0.5

for t in range(1, 8760):
    prev = soc_rfnbo[t - 1]
    if prices[t] < 20 and prev < capacity - 0.1:
        c = min(max_power, (capacity - prev) / sqrt_eff)
        charge_rfnbo[t] = c
        soc_rfnbo[t] = prev + c * sqrt_eff
    elif prices[t] > 70 and prev > 0.5:
        d = min(max_power, prev * sqrt_eff)
        discharge_rfnbo[t] = d
        soc_rfnbo[t] = prev - d / sqrt_eff
    else:
        soc_rfnbo[t] = prev

rev_rfnbo    = prices * (discharge_rfnbo * sqrt_eff - charge_rfnbo / sqrt_eff)
cumrev_rfnbo = np.cumsum(rev_rfnbo)
cumrev_std   = np.cumsum(revenue_per_hour)

monthly_std   = []
monthly_rfnbo = []
start = 0
for h in month_hours:
    monthly_std.append(revenue_per_hour[start:start+h].sum())
    monthly_rfnbo.append(rev_rfnbo[start:start+h].sum())
    start += h

fig, axes = card_fig(1, 3, (15, 4.5), "RFNBO Additionality Compliance — Impact Analysis")
fig.subplots_adjust(wspace=0.38, left=0.07, right=0.97, top=0.88, bottom=0.18)

# Panel 1: Cumulative revenue
ax = axes[0]
ax.plot(hours, cumrev_std   / 1000, color=PURPLE, linewidth=1.5, label="Unconstrained")
ax.plot(hours, cumrev_rfnbo / 1000, color=GREEN,  linewidth=1.5, linestyle="--", label="RFNBO compliant")
ax.fill_between(hours, cumrev_std / 1000, cumrev_rfnbo / 1000,
                alpha=0.12, color=RED, label="Revenue penalty")
ax.set_xlabel("Hour of year", color=MGRAY, fontsize=8)
ax.set_ylabel("Cumulative Revenue (k€)", color=MGRAY, fontsize=8)
ax.set_title("Cumulative Revenue", fontsize=9, color=DPURPLE, fontweight="bold", loc="left")
ax.legend(fontsize=7.5)

# Panel 2: Monthly comparison
ax = axes[1]
x = np.arange(len(month_names))
w = 0.38
ax.bar(x - w/2, [r/1000 for r in monthly_std],   width=w, color=PURPLE, alpha=0.85, label="Unconstrained")
ax.bar(x + w/2, [r/1000 for r in monthly_rfnbo],  width=w, color=GREEN,  alpha=0.85, label="RFNBO compliant")
ax.set_xticks(x)
ax.set_xticklabels(month_names, fontsize=7.5, color=MGRAY)
ax.set_ylabel("Revenue (k€)", color=MGRAY, fontsize=8)
ax.set_title("Monthly Comparison", fontsize=9, color=DPURPLE, fontweight="bold", loc="left")
ax.legend(fontsize=7.5)

# Panel 3: Blocked charging by hour
ax = axes[2]
rfnbo_threshold = 20.0
blocked   = (prices > rfnbo_threshold) & (charge > 0.01)
unblocked = (prices <= rfnbo_threshold) & (charge > 0.01)
hour_of_day_arr = hours % 24
blocked_counts   = np.bincount(hour_of_day_arr[blocked],   minlength=24)
unblocked_counts = np.bincount(hour_of_day_arr[unblocked], minlength=24)

ax.bar(range(24), unblocked_counts, color=GREEN, alpha=0.85, label="Allowed (≤20 €/MWh)")
ax.bar(range(24), blocked_counts, bottom=unblocked_counts, color=RED, alpha=0.75, label="Blocked by RFNBO gate")
ax.set_xlabel("Hour of day", color=MGRAY, fontsize=8)
ax.set_ylabel("Charge events (h/year)", color=MGRAY, fontsize=8)
ax.set_title("RFNBO Gate — Blocked Charging by Hour",
             fontsize=9, color=DPURPLE, fontweight="bold", loc="left")
ax.legend(fontsize=7.5)

penalty     = cumrev_std[-1] - cumrev_rfnbo[-1]
penalty_pct = penalty / cumrev_std[-1] * 100
fig.text(0.5, 0.02,
         f"Revenue penalty from RFNBO compliance: €{penalty:,.0f}  ({penalty_pct:.1f}%)",
         ha="center", fontsize=10, color=RED, fontweight="bold")

path4 = OUT_DIR / "rfnbo_analysis.png"
fig.savefig(path4, bbox_inches="tight", facecolor=LGRAY)
plt.close()
print(f"Saved: {path4}")

print("\nAll plots saved to docs/images/")
print(f"Annual revenue (simulated): €{cumulative_rev[-1]:,.0f}")
print(f"RFNBO penalty: €{penalty:,.0f} ({penalty_pct:.1f}%)")