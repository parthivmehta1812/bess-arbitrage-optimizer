"""
Example 3 — RFNBO additionality compliance
===========================================
Activates the RFNBO gate that blocks charging during hours where the
electricity price exceeds 20 EUR/MWh.  This is a proxy for ensuring the
battery only charges from renewable surplus (additionality criterion).
"""

from bess_arbitrage import optimize_bess_arbitrage

# Without RFNBO constraint
result_std = optimize_bess_arbitrage(
    battery_capacity_mwh=8.0,
    max_power_mw=2.0,
    rfnbo_compliant=False,
)

# With RFNBO constraint
result_rfnbo = optimize_bess_arbitrage(
    battery_capacity_mwh=8.0,
    max_power_mw=2.0,
    rfnbo_compliant=True,
)

rev_std   = result_std["kpis"]["annual_revenue_eur"]
rev_rfnbo = result_rfnbo["kpis"]["annual_revenue_eur"]
penalty   = rev_std - rev_rfnbo

print("\n── RFNBO compliance impact ────────────────────────────")
print(f"  Revenue (unconstrained) : €{rev_std:,.0f}")
print(f"  Revenue (RFNBO gated)   : €{rev_rfnbo:,.0f}")
print(f"  Revenue penalty         : €{penalty:,.0f}  ({penalty / rev_std * 100:.1f} %)")