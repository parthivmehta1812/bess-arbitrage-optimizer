"""
bess-arbitrage-optimizer
~~~~~~~~~~~~~~~~~~~~~~~~
MILP-based battery energy storage (BESS) arbitrage optimizer.
"""

from .loader import load_prices
from .optimizer import optimize_bess_arbitrage, optimize_peak_shaving, make_industrial_load
from .forecaster import train_price_forecaster, forecast_prices

__version__ = "0.1.0"
__all__ = [
    "optimize_bess_arbitrage",
    "optimize_peak_shaving",
    "make_industrial_load",
    "load_prices",
    "train_price_forecaster",
    "forecast_prices",
]
