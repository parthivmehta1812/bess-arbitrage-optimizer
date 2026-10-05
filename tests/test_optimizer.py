"""
Unit and integration tests for the BESS arbitrage optimizer.

Tests use a tiny synthetic price series to keep solve times under a second.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from bess_arbitrage.loader import load_prices
from bess_arbitrage.optimizer import optimize_bess_arbitrage


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_price_csv(pattern: str = "alternating") -> bytes:
    """Return a minimal CSV with 8760 hourly prices."""
    import io
    import csv

    if pattern == "alternating":
        # Simple 2-value cycle: cheap hour then expensive hour
        prices = [10.0 if i % 2 == 0 else 80.0 for i in range(8760)]
    elif pattern == "flat":
        prices = [50.0] * 8760
    else:
        rng = np.random.default_rng(42)
        prices = rng.uniform(0, 200, 8760).tolist()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["price_eur_mwh"])
    for p in prices:
        writer.writerow([p])
    return buf.getvalue().encode()


# ── Loader tests ──────────────────────────────────────────────────────────────

class TestLoader:
    def test_returns_8760_values(self):
        prices = load_prices(_make_price_csv())
        assert len(prices) == 8760

    def test_dtype_is_float64(self):
        prices = load_prices(_make_price_csv())
        assert prices.dtype == np.float64

    def test_flat_prices_preserved(self):
        prices = load_prices(_make_price_csv("flat"))
        assert np.allclose(prices, 50.0)

    def test_short_file_is_padded(self):
        import io, csv
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["price"])
        for _ in range(100):
            writer.writerow([30.0])
        prices = load_prices(buf.getvalue().encode())
        assert len(prices) == 8760
        assert prices[-1] == 30.0

    def test_no_numeric_columns_raises(self):
        csv_bytes = b"label\nfoo\nbar\n"
        with pytest.raises(ValueError, match="no numeric columns"):
            load_prices(csv_bytes)


# ── Optimizer tests ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def basic_result():
    """Solve a small fixed-sizing problem once and reuse across tests."""
    price_bytes = _make_price_csv("alternating")
    return optimize_bess_arbitrage(
        price_bytes=price_bytes,
        battery_capacity_mwh=4.0,
        max_power_mw=2.0,
        efficiency=1.0,   # perfect efficiency simplifies assertions
        optimize_sizing=False,
        rfnbo_compliant=False,
    )


class TestFixedSizing:
    def test_status_optimal(self, basic_result):
        assert basic_result["status"] in ("optimal", "feasible")

    def test_positive_revenue(self, basic_result):
        assert basic_result["annual_revenue_eur"] > 0

    def test_schedule_keys_present(self, basic_result):
        required = {"Hour", "Price_EUR_MWh", "Charge_MW", "Discharge_MW", "SoC_MWh", "Revenue_EUR"}
        assert required.issubset(basic_result["schedule"].keys())

    def test_schedule_length(self, basic_result):
        assert len(basic_result["schedule"]["Hour"]) == 8760

    def test_kpis_present(self, basic_result):
        required = {
            "annual_revenue_eur", "optimal_capacity_mwh", "optimal_power_mw",
            "ep_ratio_h", "annual_cycles",
        }
        assert required.issubset(basic_result["kpis"].keys())

    def test_no_simultaneous_charge_discharge(self, basic_result):
        s = basic_result["schedule"]
        charge    = np.array(s["Charge_MW"])
        discharge = np.array(s["Discharge_MW"])
        # Both non-zero at the same hour = violation
        both_active = (charge > 1e-4) & (discharge > 1e-4)
        assert not both_active.any(), "Simultaneous charge + discharge detected."

    def test_soc_within_capacity(self, basic_result):
        soc = np.array(basic_result["schedule"]["SoC_MWh"])
        cap = basic_result["optimal_capacity_mwh"]
        assert (soc >= -1e-6).all()
        assert (soc <= cap + 1e-4).all()

    def test_power_limits_respected(self, basic_result):
        s   = basic_result["schedule"]
        pwr = basic_result["optimal_power_mw"]
        assert max(s["Charge_MW"])    <= pwr + 1e-4
        assert max(s["Discharge_MW"]) <= pwr + 1e-4

    def test_capacity_unchanged(self, basic_result):
        assert math.isclose(basic_result["optimal_capacity_mwh"], 4.0, rel_tol=1e-6)

    def test_power_unchanged(self, basic_result):
        assert math.isclose(basic_result["optimal_power_mw"], 2.0, rel_tol=1e-6)


class TestRfnboGate:
    def test_no_charging_above_threshold(self):
        price_bytes = _make_price_csv("alternating")  # prices are 10 and 80
        result = optimize_bess_arbitrage(
            price_bytes=price_bytes,
            battery_capacity_mwh=4.0,
            max_power_mw=2.0,
            efficiency=1.0,
            rfnbo_compliant=True,
        )
        prices  = np.array(result["schedule"]["Price_EUR_MWh"])
        charges = np.array(result["schedule"]["Charge_MW"])
        # Charging must be zero whenever price > 20 EUR/MWh
        forbidden = (prices > 20.0) & (charges > 1e-4)
        assert not forbidden.any(), "RFNBO gate violated: charging above 20 EUR/MWh."


class TestMissingPyomo:
    def test_import_error_raised(self, monkeypatch):
        import bess_arbitrage.optimizer as mod
        monkeypatch.setattr(mod, "_PYOMO_OK", False)
        with pytest.raises(ImportError, match="Pyomo"):
            mod.optimize_bess_arbitrage(price_bytes=_make_price_csv())
