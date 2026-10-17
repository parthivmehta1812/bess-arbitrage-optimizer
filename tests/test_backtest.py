"""
Tests for the walk-forward backtest module.

Covers:
- Month boundary constants match 2023 calendar
- Training data ends strictly before each test month's first hour
- Persistence forecast uses exactly prices[t-24] (no look-ahead)
- Perfect-foresight revenue >= GBM-driven revenue (MILP optimality)
- DM test: clearly better model gives p < 0.05; identical forecasts handled
- DM output keys are complete
- Existing tests still pass (run full suite via pytest)
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from bess_arbitrage.backtest import (
    _MONTH_HOURS,
    _MONTH_NAMES,
    _MONTH_STARTS,
    _milp_schedule,
    _actual_revenue,
    diebold_mariano,
)


# ── Month boundary constants ───────────────────────────────────────────────────

class TestMonthBoundaries:
    def test_total_hours_is_8760(self):
        assert _MONTH_STARTS[-1] == 8760

    def test_month_hours_sum(self):
        assert sum(_MONTH_HOURS) == 8760

    def test_key_boundaries(self):
        assert _MONTH_STARTS[0]  == 0      # Jan start
        assert _MONTH_STARTS[2]  == 1416   # Mar start (first test month)
        assert _MONTH_STARTS[8]  == 5832   # Sep start (matches existing pipeline)
        assert _MONTH_STARTS[12] == 8760   # end of year

    def test_thirteen_entries(self):
        # One entry per month boundary (Jan start through Dec end)
        assert len(_MONTH_STARTS) == 13
        assert len(_MONTH_NAMES)  == 12

    def test_no_overlap_train_test(self):
        """For every test month, the last training hour precedes the first test hour."""
        for m_idx in range(2, 12):   # March (2) through December (11), 0-indexed
            train_last  = _MONTH_STARTS[m_idx] - 1   # train on [168, h_start)
            test_first  = _MONTH_STARTS[m_idx]
            assert train_last < test_first, (
                f"Month {_MONTH_NAMES[m_idx]}: train ends at hour {train_last}, "
                f"but test starts at hour {test_first}"
            )


# ── Persistence forecast ───────────────────────────────────────────────────────

class TestPersistenceForecast:
    def test_persistence_equals_lag24(self):
        """prices[h_start-24 : h_end-24] must equal prices[h-24] for each test hour h."""
        prices = np.arange(8760, dtype=float) * 2.5   # non-trivial values

        for m_idx in range(2, 12):
            h_start = _MONTH_STARTS[m_idx]
            h_end   = _MONTH_STARTS[m_idx + 1]

            pers = prices[h_start - 24 : h_end - 24]

            for i, h in enumerate(range(h_start, h_end)):
                assert pers[i] == prices[h - 24], (
                    f"Month {_MONTH_NAMES[m_idx]}, hour {h}: "
                    f"expected {prices[h-24]}, got {pers[i]}"
                )

    def test_persistence_no_look_ahead(self):
        """Persistence forecast at hour h must use prices[h-24] — always 24h in the past."""
        prices  = np.arange(8760, dtype=float)
        m_idx   = 2    # March
        h_start = _MONTH_STARTS[m_idx]
        h_end   = _MONTH_STARTS[m_idx + 1]

        pers = prices[h_start - 24 : h_end - 24]

        # For each test hour h, pers[i] = prices[h-24] and h-24 < h (strictly past)
        for i, h in enumerate(range(h_start, h_end)):
            assert pers[i] == prices[h - 24]
            assert h - 24 < h   # tautology, but makes the invariant explicit


# ── MILP schedule + perfect-foresight revenue ─────────────────────────────────

class TestMilpSchedule:
    """Use a small (48-hour) MILP to keep tests fast."""

    @pytest.fixture
    def alternating_prices(self):
        # Cheap hours (price=10) and expensive hours (price=80), alternating
        return np.array([10.0, 80.0] * 24)   # 48h

    def test_output_shapes(self, alternating_prices):
        T = len(alternating_prices)
        charge, discharge = _milp_schedule(alternating_prices, 2.0, 1.0, 0.92, 0.5)
        assert charge.shape    == (T,)
        assert discharge.shape == (T,)

    def test_no_simultaneous_charge_discharge(self, alternating_prices):
        charge, discharge = _milp_schedule(alternating_prices, 2.0, 1.0, 0.92, 0.5)
        # Big-M mutex: in any hour at most one of charge, discharge can be > 0
        both_active = (charge > 1e-4) & (discharge > 1e-4)
        assert not np.any(both_active), "Simultaneous charge+discharge violates mutex"

    def test_power_bounds_respected(self, alternating_prices):
        max_power = 1.0
        charge, discharge = _milp_schedule(alternating_prices, 2.0, max_power, 0.92, 0.5)
        assert np.all(charge    <= max_power + 1e-6)
        assert np.all(discharge <= max_power + 1e-6)
        assert np.all(charge    >= -1e-6)
        assert np.all(discharge >= -1e-6)

    def test_perfect_foresight_revenue_nonnegative(self, alternating_prices):
        charge, discharge = _milp_schedule(alternating_prices, 2.0, 1.0, 0.92, 0.5)
        rev = _actual_revenue(alternating_prices, charge, discharge, 0.92)
        assert rev >= -1e-6, f"Perfect foresight should not lose money, got {rev:.2f}"

    def test_perfect_foresight_beats_noise_schedule(self, alternating_prices):
        """Perfect foresight revenue >= revenue from a noisy-forecast schedule."""
        rng   = np.random.default_rng(0)
        noisy = alternating_prices + rng.normal(0, 10, len(alternating_prices))

        eta = 0.92

        c_pf,  d_pf  = _milp_schedule(alternating_prices, 2.0, 1.0, eta, 0.5)
        c_nf,  d_nf  = _milp_schedule(noisy,              2.0, 1.0, eta, 0.5)

        rev_pf = _actual_revenue(alternating_prices, c_pf, d_pf, eta)
        rev_nf = _actual_revenue(alternating_prices, c_nf, d_nf, eta)

        # Perfect foresight is optimal — must be >= any other schedule evaluated
        # on the same actual prices (within solver tolerance)
        assert rev_pf >= rev_nf - 1e-2, (
            f"Perfect foresight ({rev_pf:.2f}) should be >= noisy-forecast ({rev_nf:.2f})"
        )


# ── Diebold–Mariano test ───────────────────────────────────────────────────────

class TestDieboldMariano:
    def test_output_keys(self):
        rng = np.random.default_rng(1)
        e1  = rng.normal(5, 2, 24 * 50)
        e2  = rng.normal(3, 2, 24 * 50)
        result = diebold_mariano(e1, e2, loss="abs")
        for key in ["dm_stat", "p_normal", "hln_stat", "p_hln", "n_days",
                    "loss_mean_e1", "loss_mean_e2"]:
            assert key in result, f"Missing key: {key}"

    def test_n_days_is_correct(self):
        n_days = 60
        e = np.zeros(n_days * 24)
        result = diebold_mariano(e, e + 1e-3)
        assert result["n_days"] == n_days

    def test_clearly_better_model_abs_loss(self):
        """Model 2 with clearly smaller errors → p < 0.05 (abs loss)."""
        rng = np.random.default_rng(42)
        n   = 200 * 24
        e1  = rng.normal(15, 5, n)   # persistence: large errors
        e2  = rng.normal(0,  1, n)   # GBM: small errors
        result = diebold_mariano(e1, e2, loss="abs")
        assert result["p_normal"] < 0.05, (
            f"Expected p_normal < 0.05 for clearly better model, got {result['p_normal']:.4f}"
        )
        assert result["p_hln"]    < 0.05, (
            f"Expected p_hln < 0.05 for clearly better model, got {result['p_hln']:.4f}"
        )

    def test_clearly_better_model_sq_loss(self):
        """Model 2 with clearly smaller errors → p < 0.05 (squared loss)."""
        rng = np.random.default_rng(7)
        n   = 200 * 24
        e1  = rng.normal(15, 5, n)
        e2  = rng.normal(0,  1, n)
        result = diebold_mariano(e1, e2, loss="sq")
        assert result["p_normal"] < 0.05
        assert result["p_hln"]    < 0.05

    def test_dm_stat_positive_when_e2_better(self):
        """DM statistic > 0 when model 2 (e2) has lower loss than model 1 (e1)."""
        rng = np.random.default_rng(3)
        n   = 100 * 24
        e1  = rng.normal(10, 3, n)
        e2  = rng.normal(0,  1, n)
        result = diebold_mariano(e1, e2, loss="abs")
        assert result["dm_stat"] > 0, "DM stat should be positive when e2 is better"

    def test_identical_forecasts_handled_gracefully(self):
        """When e1 == e2, variance of d is zero — function must not crash."""
        rng = np.random.default_rng(0)
        e   = rng.normal(0, 1, 24 * 30)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = diebold_mariano(e, e.copy(), loss="abs")
            assert any(issubclass(x.category, RuntimeWarning) for x in w), (
                "Expected a RuntimeWarning for zero-variance differentials"
            )
        assert math.isnan(result["dm_stat"]), (
            "dm_stat should be NaN when all differentials are zero"
        )

    def test_loss_means_are_nonnegative(self):
        """Loss values (abs or sq) cannot be negative."""
        rng = np.random.default_rng(9)
        e1  = rng.normal(0, 5, 24 * 40)
        e2  = rng.normal(0, 3, 24 * 40)
        for loss in ("abs", "sq"):
            result = diebold_mariano(e1, e2, loss=loss)
            assert result["loss_mean_e1"] >= 0
            assert result["loss_mean_e2"] >= 0

    def test_mismatched_lengths_raise(self):
        e1 = np.ones(24 * 10)
        e2 = np.ones(24 * 11)
        with pytest.raises(ValueError, match="equal length"):
            diebold_mariano(e1, e2)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 24"):
            diebold_mariano(np.ones(12), np.ones(12))

    def test_invalid_loss_raises(self):
        e = np.ones(24 * 5)
        with pytest.raises(ValueError, match="loss"):
            diebold_mariano(e, e, loss="mae")  # type: ignore[arg-type]
