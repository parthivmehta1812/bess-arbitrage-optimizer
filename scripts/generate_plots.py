"""
Generate sample result plots for the README.
Run once after installing dependencies:
    pip install matplotlib seaborn pandas numpy
    python scripts/generate_plots.py
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Synthetic price + schedule data (mirrors realistic MILP output) ───────────
rng = np.random.default_rng(42)
hours = np.arange(8760)

# Realistic price shape: low at night, high at peak
hour_of_day = hours % 24
base_price = 35 + 55 * np.clip(np.sin(np.pi * (hour_of_day - 6) / 14), 0, 1)
seasonal   = 10 * np.cos(2 * np.pi * hours / 8760)
noise      = rng.normal(0, 8, 8760)
prices     = np.clip(base_price + seasonal + noise, -10, 200)

# Simulate charge when price < 30, discharge when price > 70
capacity   = 8.0   # MWh
max_power  = 2.0   # MW
efficiency = 0.92
sqrt_eff   = math.sqrt(efficiency)
soc        = np.zeros(8760)
charge     = np.zeros(8760)
discharge  = np.zeros(8760)
soc[0]     = capacity * 0.5

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

# ── Plot styling ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "figure.dpi":      150,
})
BLUE   = "#2563EB"
GREEN  = "#16A34A"
RED    = "#DC2626"
ORANGE = "#EA580C"
GRAY   = "#6B7280"

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Weekly dispatch snapshot (first 2 weeks)
# ─────────────────────────────────────────────────────────────────────────────
WINDOW = 24 * 14  # two weeks
t_w    = np.arange(WINDOW)

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle("BESS Arbitrage — 2-Week Dispatch Snapshot", fontsize=14, fontweight="bold", y=0.98)

# Price
ax = axes[0]
ax.plot(t_w, prices[:WINDOW], color=GRAY, linewidth=1, label="Day-ahead price")
ax.axhline(30, color=BLUE,  linewidth=0.8, linestyle="--", alpha=0.7, label="Charge threshold (30 €)")
ax.axhline(70, color=RED,   linewidth=0.8, linestyle="--", alpha=0.7, label="Discharge threshold (70 €)")
ax.set_ylabel("Price (EUR/MWh)")
ax.legend(fontsize=8, loc="upper right")

# Charge / discharge
ax = axes[1]
ax.bar(t_w, charge[:WINDOW],    color=BLUE,  alpha=0.8, label="Charge (MW)",    width=1)
ax.bar(t_w, -discharge[:WINDOW], color=RED,  alpha=0.8, label="Discharge (MW)", width=1)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Power (MW)")
ax.legend(fontsize=8, loc="upper right")

# SoC
ax = axes[2]
ax.fill_between(t_w, soc[:WINDOW], alpha=0.3, color=GREEN)
ax.plot(t_w, soc[:WINDOW], color=GREEN, linewidth=1.2, label="State of Charge")
ax.axhline(capacity, color=GRAY, linewidth=0.8, linestyle=":", label=f"Capacity ({capacity} MWh)")
ax.set_ylabel("SoC (MWh)")
ax.set_xlabel("Hour of year")
ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
path1 = OUT_DIR / "dispatch_snapshot.png"
fig.savefig(path1, bbox_inches="tight")
plt.close()
print(f"Saved: {path1}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Annual cumulative revenue + price distribution
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Annual Performance Summary", fontsize=14, fontweight="bold")

# Cumulative revenue
ax = axes[0]
ax.plot(hours, cumulative_rev / 1000, color=GREEN, linewidth=1.5)
ax.fill_between(hours, cumulative_rev / 1000, alpha=0.15, color=GREEN)
ax.set_xlabel("Hour of year")
ax.set_ylabel("Cumulative Revenue (k€)")
ax.set_title(f"Total: €{cumulative_rev[-1]:,.0f}")

# Price histogram with charge/discharge zones
ax = axes[1]
ax.hist(prices, bins=80, color=GRAY, alpha=0.6, label="All hours")
charge_prices    = prices[charge > 0.01]
discharge_prices = prices[discharge > 0.01]
ax.hist(charge_prices,    bins=40, color=BLUE, alpha=0.7, label="Charge hours")
ax.hist(discharge_prices, bins=40, color=RED,  alpha=0.7, label="Discharge hours")
ax.set_xlabel("Price (EUR/MWh)")
ax.set_ylabel("Hours")
ax.set_title("Price Distribution by Operation Mode")
ax.legend(fontsize=8)

plt.tight_layout()
path2 = OUT_DIR / "annual_summary.png"
fig.savefig(path2, bbox_inches="tight")
plt.close()
print(f"Saved: {path2}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Monthly revenue bar chart
# ─────────────────────────────────────────────────────────────────────────────
month_hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
monthly_rev = []
start = 0
for h in month_hours:
    monthly_rev.append(revenue_per_hour[start:start+h].sum())
    start += h

fig, ax = plt.subplots(figsize=(10, 4))
colors = [GREEN if r > 0 else RED for r in monthly_rev]
bars = ax.bar(month_names, [r/1000 for r in monthly_rev], color=colors, alpha=0.85, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Revenue (k€)")
ax.set_title("Monthly Arbitrage Revenue", fontsize=13, fontweight="bold")
for bar, val in zip(bars, monthly_rev):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"€{val:,.0f}", ha="center", va="bottom", fontsize=7.5)
plt.tight_layout()
path3 = OUT_DIR / "monthly_revenue.png"
fig.savefig(path3, bbox_inches="tight")
plt.close()
print(f"Saved: {path3}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — RFNBO compliance comparison
# ─────────────────────────────────────────────────────────────────────────────

# Simulate RFNBO-gated schedule: block charging when price > 20 EUR/MWh
soc_rfnbo      = np.zeros(8760)
charge_rfnbo   = np.zeros(8760)
discharge_rfnbo = np.zeros(8760)
soc_rfnbo[0]  = capacity * 0.5

for t in range(1, 8760):
    prev = soc_rfnbo[t - 1]
    if prices[t] < 20 and prev < capacity - 0.1:          # RFNBO gate: ≤20 only
        c = min(max_power, (capacity - prev) / sqrt_eff)
        charge_rfnbo[t] = c
        soc_rfnbo[t] = prev + c * sqrt_eff
    elif prices[t] > 70 and prev > 0.5:
        d = min(max_power, prev * sqrt_eff)
        discharge_rfnbo[t] = d
        soc_rfnbo[t] = prev - d / sqrt_eff
    else:
        soc_rfnbo[t] = prev

rev_rfnbo     = prices * (discharge_rfnbo * sqrt_eff - charge_rfnbo / sqrt_eff)
cumrev_rfnbo  = np.cumsum(rev_rfnbo)
cumrev_std    = np.cumsum(revenue_per_hour)

# Monthly breakdown for both
monthly_std   = []
monthly_rfnbo = []
start = 0
for h in month_hours:
    monthly_std.append(revenue_per_hour[start:start+h].sum())
    monthly_rfnbo.append(rev_rfnbo[start:start+h].sum())
    start += h

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("RFNBO Additionality Compliance — Impact Analysis", fontsize=13, fontweight="bold")

# Panel 1: Cumulative revenue comparison
ax = axes[0]
ax.plot(hours, cumrev_std   / 1000, color=GREEN, linewidth=1.5, label="Unconstrained")
ax.plot(hours, cumrev_rfnbo / 1000, color=ORANGE, linewidth=1.5, linestyle="--", label="RFNBO compliant")
ax.fill_between(hours,
                cumrev_std / 1000,
                cumrev_rfnbo / 1000,
                alpha=0.15, color=RED, label="Revenue penalty")
ax.set_xlabel("Hour of year")
ax.set_ylabel("Cumulative Revenue (k€)")
ax.set_title("Cumulative Revenue")
ax.legend(fontsize=8)

# Panel 2: Monthly comparison
ax = axes[1]
x = np.arange(len(month_names))
w = 0.38
ax.bar(x - w/2, [r/1000 for r in monthly_std],   width=w, color=GREEN,  alpha=0.85, label="Unconstrained")
ax.bar(x + w/2, [r/1000 for r in monthly_rfnbo],  width=w, color=ORANGE, alpha=0.85, label="RFNBO compliant")
ax.set_xticks(x)
ax.set_xticklabels(month_names, fontsize=8)
ax.set_ylabel("Revenue (k€)")
ax.set_title("Monthly Comparison")
ax.legend(fontsize=8)

# Panel 3: Charging hours blocked by RFNBO gate
ax = axes[2]
rfnbo_threshold = 20.0
blocked   = (prices > rfnbo_threshold) & (charge > 0.01)
unblocked = (prices <= rfnbo_threshold) & (charge > 0.01)
hour_of_day_arr = hours % 24

blocked_counts   = np.bincount(hour_of_day_arr[blocked],   minlength=24)
unblocked_counts = np.bincount(hour_of_day_arr[unblocked], minlength=24)

ax.bar(range(24), unblocked_counts, color=GREEN,  alpha=0.85, label="Allowed (≤20 €/MWh)")
ax.bar(range(24), blocked_counts,   bottom=unblocked_counts,
       color=RED, alpha=0.75, label="Blocked by RFNBO gate")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Charge events (hours/year)")
ax.set_title("RFNBO Gate — Blocked Charging by Hour")
ax.legend(fontsize=8)

penalty     = cumrev_std[-1] - cumrev_rfnbo[-1]
penalty_pct = penalty / cumrev_std[-1] * 100
fig.text(0.5, -0.02,
         f"Revenue penalty from RFNBO compliance: €{penalty:,.0f}  ({penalty_pct:.1f}%)",
         ha="center", fontsize=10, color=RED, fontweight="bold")

plt.tight_layout()
path4 = OUT_DIR / "rfnbo_analysis.png"
fig.savefig(path4, bbox_inches="tight")
plt.close()
print(f"Saved: {path4}")

print("\nAll plots saved to docs/images/")
print("Annual revenue (simulated):", f"€{cumulative_rev[-1]:,.0f}")
print(f"RFNBO penalty: €{penalty:,.0f} ({penalty_pct:.1f}%)")