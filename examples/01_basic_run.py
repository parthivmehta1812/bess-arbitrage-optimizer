"""
Example 1 — Basic fixed-sizing run
===================================
Runs the optimizer with a fixed 8 MWh / 2 MW battery configuration
using the bundled 2024 German ENTSO-E price sample.
"""

from bess_arbitrage import optimize_bess_arbitrage

result = optimize_bess_arbitrage(
    battery_capacity_mwh=8.0,
    max_power_mw=2.0,
    efficiency=0.92,
)

kpis = result["kpis"]
print("\n── Results ─────────────────────────────────────────────")
print(f"  Annual revenue   : €{kpis['annual_revenue_eur']:,.0f}")
print(f"  Capacity         : {kpis['optimal_capacity_mwh']} MWh")
print(f"  Power rating     : {kpis['optimal_power_mw']} MW")
print(f"  E:P ratio        : {kpis['ep_ratio_h']} h")
print(f"  Annual cycles    : {kpis['annual_cycles']}")
print(f"  Avg charge price : {kpis['avg_charge_price_eur_mwh']} EUR/MWh")
print(f"  Avg disch. price : {kpis['avg_discharge_price_eur_mwh']} EUR/MWh")
print(f"  Solver status    : {kpis['solver_status']}")