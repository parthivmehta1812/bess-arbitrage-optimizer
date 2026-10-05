"""
Example 2 — Co-optimised sizing
================================
Lets the optimizer jointly select the best battery capacity (MWh) AND
power rating (MW) subject to a 2–4 h E:P ratio window.
"""

from bess_arbitrage import optimize_bess_arbitrage

result = optimize_bess_arbitrage(
    optimize_sizing=True,
    max_grid_mw=10.0,
    efficiency=0.92,
    ep_ratio_min=2.0,
    ep_ratio_max=4.0,
)

kpis = result["kpis"]
print("\n── Co-optimised sizing results ─────────────────────────")
print(f"  Optimal capacity : {kpis['optimal_capacity_mwh']} MWh")
print(f"  Optimal power    : {kpis['optimal_power_mw']} MW")
print(f"  E:P ratio        : {kpis['ep_ratio_h']} h")
print(f"  Annual revenue   : €{kpis['annual_revenue_eur']:,.0f}")
print(f"  Annual cycles    : {kpis['annual_cycles']}")