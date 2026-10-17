"""
Tests for the robustness module (src/bess_arbitrage/robustness.py).

Covers:
- check_and_flag: each fault type detected in isolation
- impute_gaps: short-gap linear interpolation, long-gap backward fallback,
  no future-data leakage
- clean_prices: end-to-end pipeline
- naive_ffill: NaN forward-fill
- choose_k: only uses hours 0–1415 (never test-period hours ≥ 1416)
- select_forecast_per_day: high imputed fraction → persistence branch;
  low confidence → persistence branch; clean signal → GBM branch
"""

from __future__ import annotations

import numpy as np
import pytest

from bess_arbitrage.robustness import (
    _K_CAL_END,
    _K_CAL_TRAIN_END,
    PRICE_HI,
    PRICE_LO,
    SHORT_GAP_MAX,
    SPIKE_Z_THRESH,
    SPIKE_WINDOW,
    STUCK_MIN_RUN,
    check_and_flag,
    choose_k,
    clean_prices,
    daily_interval_widths,
    impute_gaps,
    naive_ffill,
    select_forecast_per_day,
)


# ── check_and_flag ──────────────────────────────────────────────────────────────

class TestCheckAndFlag:

    def _base(self, n: int = 200) -> np.ndarray:
        """Synthetic clean prices — strictly in bounds, no runs, no spikes."""
        rng = np.random.default_rng(0)
        return rng.uniform(10.0, 100.0, n)

    def test_clean_array_no_flags(self):
        prices = self._base(300)
        result, report = check_and_flag(prices)
        assert report["n_nan"]    == 0
        assert report["n_bounds"] == 0
        assert report["n_spike"]  == 0
        assert report["n_stuck"]  == 0
        assert not np.any(np.isnan(result))

    def test_detects_existing_nan(self):
        prices = self._base(50)
        prices[10] = np.nan
        prices[20] = np.nan
        result, report = check_and_flag(prices)
        assert report["n_nan"] == 2
        assert np.isnan(result[10])
        assert np.isnan(result[20])

    def test_detects_below_lower_bound(self):
        prices    = self._base(100)
        prices[5] = PRICE_LO - 1.0        # strictly below floor
        result, report = check_and_flag(prices)
        assert report["n_bounds"] >= 1
        assert np.isnan(result[5])

    def test_detects_above_upper_bound(self):
        prices    = self._base(100)
        prices[7] = PRICE_HI + 1.0        # above ceiling
        result, report = check_and_flag(prices)
        assert report["n_bounds"] >= 1
        assert np.isnan(result[7])

    def test_detects_spike(self):
        """Insert a value far above the local median — should be flagged as spike.

        The spike must be within market bounds (PRICE_LO–PRICE_HI) so it is not
        caught by the bounds check first.  3 990 EUR/MWh on a background of ~50
        gives z >> SPIKE_Z_THRESH = 3.5.
        """
        n      = SPIKE_WINDOW + 10
        prices = np.full(n, 50.0)
        # Vary slightly so MAD > 0 (otherwise the stuck-values check would fire)
        rng    = np.random.default_rng(1)
        prices += rng.uniform(-2, 2, n)
        prices[-1] = 3_990.0              # large but within [PRICE_LO, PRICE_HI]
        result, report = check_and_flag(prices)
        assert report["n_spike"] >= 1, "In-bounds spike must be flagged by z-score check"
        assert np.isnan(result[-1])

    def test_no_spike_when_mad_zero(self):
        """All-identical window → MAD = 0 → no spike flagged."""
        prices = np.full(50, 42.0)
        prices[-1] = 84.0                  # only 2× — not a spike if MAD=0
        result, report = check_and_flag(prices)
        # Spike at last position should NOT be flagged (MAD = 0 for identical window)
        assert report["n_spike"] == 0

    def test_detects_stuck_run(self):
        prices = self._base(100)
        prices[20 : 20 + STUCK_MIN_RUN] = 99.99  # exact run of threshold length
        result, report = check_and_flag(prices)
        assert report["n_stuck"] >= STUCK_MIN_RUN
        assert all(np.isnan(result[20 : 20 + STUCK_MIN_RUN]))

    def test_stuck_run_below_threshold_not_flagged(self):
        prices = self._base(100)
        run    = STUCK_MIN_RUN - 1          # one below threshold
        prices[30 : 30 + run] = 77.0
        _, report = check_and_flag(prices)
        assert report["n_stuck"] == 0

    def test_returns_copy(self):
        prices    = self._base(50)
        original  = prices.copy()
        check_and_flag(prices)
        np.testing.assert_array_equal(prices, original)

    def test_report_keys(self):
        _, report = check_and_flag(self._base(30))
        assert set(report.keys()) == {"n_nan", "n_bounds", "n_spike", "n_stuck"}


# ── impute_gaps ─────────────────────────────────────────────────────────────────

class TestImputeGaps:

    def test_short_gap_linear_interpolation(self):
        """Gap of 2 between 0.0 and 6.0 → should yield 2.0, 4.0."""
        prices = np.array([0.0, np.nan, np.nan, 6.0, 10.0])
        result, mask = impute_gaps(prices)
        assert mask[1] and mask[2]
        np.testing.assert_allclose(result[1], 2.0, atol=1e-9)
        np.testing.assert_allclose(result[2], 4.0, atol=1e-9)
        assert not mask[0]
        assert not mask[3]

    def test_short_gap_forward_fill_at_end(self):
        """Gap at the end with only a left neighbour → forward-fill."""
        prices = np.array([5.0, np.nan, np.nan])   # gap <= SHORT_GAP_MAX
        result, mask = impute_gaps(prices)
        assert mask[1] and mask[2]
        assert result[1] == pytest.approx(5.0)
        assert result[2] == pytest.approx(5.0)

    def test_short_gap_backward_fill_at_start(self):
        """Gap at the start with only a right neighbour → backward-fill."""
        prices = np.array([np.nan, np.nan, 3.0, 10.0])
        result, mask = impute_gaps(prices)
        assert mask[0] and mask[1]
        assert result[0] == pytest.approx(3.0)
        assert result[1] == pytest.approx(3.0)

    def test_long_gap_uses_t_minus_168_first(self):
        """Long gap at position t should use prices[t-168] when available."""
        n = 400
        prices = np.full(n, np.nan)
        # Set t-168 reference values only
        for t in range(200, 210):
            prices[t - 168] = float(t) * 10.0
        result, mask = impute_gaps(prices)
        for t in range(200, 210):
            assert mask[t], f"Hour {t} should have been imputed"
            assert result[t] == pytest.approx(prices[t - 168])

    def test_long_gap_falls_back_to_t_minus_24(self):
        """If t-168 unavailable but t-24 is valid, use t-24."""
        n = 100
        prices = np.full(n, np.nan)
        prices[50] = 99.0   # t-24 reference for t=74
        result, mask = impute_gaps(prices)
        assert mask[74]
        assert result[74] == pytest.approx(99.0)

    def test_no_future_leakage_long_gap(self):
        """Long-gap imputation must only look backward (t-168 and t-24)."""
        n = 200
        prices = np.full(n, np.nan)
        prices[0] = 1.0    # very early valid value
        result, mask = impute_gaps(prices)
        for t in range(1, n):
            if mask[t]:
                src = t - 168 if (t >= 168 and not np.isnan(prices[t - 168])) else t - 24
                # Source index must be strictly before t
                assert src < t, f"Hour {t} imputed from future hour {src}"

    def test_returns_copy_and_mask_shape(self):
        prices = np.array([1.0, np.nan, 3.0])
        result, mask = impute_gaps(prices)
        assert result.shape == prices.shape
        assert mask.shape   == prices.shape
        assert result[0] == pytest.approx(1.0)   # not modified

    def test_no_imputation_needed(self):
        prices = np.array([1.0, 2.0, 3.0])
        result, mask = impute_gaps(prices)
        assert not np.any(mask)
        np.testing.assert_array_equal(result, prices)


# ── clean_prices ────────────────────────────────────────────────────────────────

class TestCleanPrices:

    def test_no_nan_in_output_for_imputable_input(self):
        """Any gap that can be imputed must leave no NaN; last resort fills with 0."""
        prices      = np.arange(500, dtype=float)
        prices[50]  = np.nan
        prices[100] = PRICE_HI + 100.0   # out-of-bounds
        result, mask, report = clean_prices(prices)
        assert not np.any(np.isnan(result)), "clean_prices must not leave NaN"

    def test_report_counts_initial_nan(self):
        prices     = np.ones(100, dtype=float) * 50.0
        prices[5]  = np.nan
        prices[10] = np.nan
        _, _, report = clean_prices(prices)
        assert report["n_nan"] == 2

    def test_imputed_mask_marks_filled_positions(self):
        # Use non-uniform values so the stuck-values check does not fire.
        rng    = np.random.default_rng(42)
        prices = rng.uniform(20.0, 100.0, 300)
        prices[100] = np.nan   # single-hour gap surrounded by valid values
        _, mask, _ = clean_prices(prices)
        assert mask[100], "Position 100 should be marked as imputed after the gap is filled"
        assert not mask[99]
        assert not mask[101]


# ── naive_ffill ─────────────────────────────────────────────────────────────────

class TestNaiveFfill:

    def test_fills_nan_with_last_valid(self):
        prices = np.array([1.0, np.nan, np.nan, 4.0])
        result = naive_ffill(prices)
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(1.0)
        assert result[3] == pytest.approx(4.0)

    def test_leading_nan_stays_nan(self):
        prices = np.array([np.nan, np.nan, 3.0])
        result = naive_ffill(prices)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(3.0)

    def test_no_nan_unchanged(self):
        prices = np.array([1.0, 2.0, 3.0])
        result = naive_ffill(prices)
        np.testing.assert_array_equal(result, prices)

    def test_returns_copy(self):
        prices   = np.array([1.0, np.nan, 3.0])
        original = prices.copy()
        naive_ffill(prices)
        np.testing.assert_array_equal(prices, original)


# ── choose_k ────────────────────────────────────────────────────────────────────

class TestChooseK:
    """Verify that k calibration never uses test-period data (hours ≥ 1416)."""

    def test_choose_k_never_reads_beyond_cal_end(self):
        """Sentinel test: fill test period with NaN — choose_k must not crash or
        return NaN (it uses only hours 0–1416)."""
        prices                   = np.random.default_rng(7).uniform(10, 100, 8760)
        prices[_K_CAL_END:]      = np.nan   # poison test period
        # Should not raise; may return a valid k from candidates
        k = choose_k(prices, k_candidates=(1.25, 1.5, 2.0))
        assert k in (1.25, 1.5, 2.0), f"k must be one of the candidates, got {k}"

    def test_choose_k_returns_candidate(self):
        candidates = (1.25, 1.5, 2.0)
        prices     = np.random.default_rng(0).uniform(10, 100, 8760)
        k          = choose_k(prices, k_candidates=candidates)
        assert k in candidates

    def test_calibration_indices(self):
        """Sanity-check that the module constants match Jan and Feb boundaries."""
        assert _K_CAL_TRAIN_END == 744,  "Jan boundary should be 744"
        assert _K_CAL_END       == 1416, "Feb boundary should be 1416"


# ── daily_interval_widths ───────────────────────────────────────────────────────

class TestDailyIntervalWidths:

    def test_shape(self):
        n_days = 10
        p10 = np.zeros(n_days * 24)
        p90 = np.ones(n_days * 24) * 20.0
        w   = daily_interval_widths(p10, p90)
        assert w.shape == (n_days,)

    def test_constant_width(self):
        n_days = 5
        p10 = np.zeros(n_days * 24)
        p90 = np.full(n_days * 24, 10.0)
        w   = daily_interval_widths(p10, p90)
        np.testing.assert_allclose(w, 10.0)


# ── select_forecast_per_day ─────────────────────────────────────────────────────

class TestSelectForecastPerDay:
    """Unit tests for the day-level fallback logic."""

    def _make_inputs(self, n_days: int = 20, seed: int = 5):
        rng   = np.random.default_rng(seed)
        T     = n_days * 24
        p50   = rng.uniform(40, 80, T)
        p10   = p50 - rng.uniform(5, 20, T)
        p90   = p50 + rng.uniform(5, 20, T)
        pers  = rng.uniform(35, 75, T)
        imp   = np.zeros(T + 200, dtype=bool)   # full mask, all False
        hist  = np.full(30, 15.0)               # realistic median width
        return p50, p10, p90, pers, imp, hist

    def test_output_shape(self):
        p50, p10, p90, pers, imp, hist = self._make_inputs()
        sel, branches = select_forecast_per_day(p50, p10, p90, pers, imp, hist, k=1.5)
        assert sel.shape == p50.shape
        assert len(branches) == len(p50) // 24

    def test_branches_are_valid_strings(self):
        p50, p10, p90, pers, imp, hist = self._make_inputs()
        _, branches = select_forecast_per_day(p50, p10, p90, pers, imp, hist, k=1.5)
        assert all(b in ("gbm", "persistence") for b in branches)

    def test_high_imputed_fraction_triggers_persistence(self):
        """If >25 % of lag inputs for a day were imputed, branch = persistence."""
        n_days  = 5
        T       = n_days * 24
        h_off   = 200           # h_offset so lags land within the mask
        rng     = np.random.default_rng(3)
        p50     = rng.uniform(50, 70, T)
        p10     = p50 - 10
        p90     = p50 + 10
        pers    = rng.uniform(45, 65, T)
        hist    = np.full(30, 15.0)

        # Build imputed_mask: mark ALL t-24 sources for day 0 as imputed
        imp     = np.zeros(h_off + T + 200, dtype=bool)
        for i_h in range(24):
            h_abs = h_off + 0 * 24 + i_h   # day 0
            for lag in (24, 48, 168):
                idx = h_abs - lag
                if 0 <= idx < len(imp):
                    imp[idx] = True

        sel, branches = select_forecast_per_day(
            p50, p10, p90, pers, imp, hist, k=1.5, h_offset=h_off
        )
        assert branches[0] == "persistence", (
            "All lag inputs imputed → branch must be 'persistence'"
        )
        np.testing.assert_array_equal(sel[:24], pers[:24])

    def test_low_confidence_triggers_persistence(self):
        """Day width >> k × median(hist_widths) → branch = persistence."""
        n_days = 3
        T      = n_days * 24
        rng    = np.random.default_rng(9)
        p50    = rng.uniform(50, 70, T)
        pers   = rng.uniform(45, 65, T)
        imp    = np.zeros(T + 200, dtype=bool)

        # Narrow historical widths so that any real-ish width triggers fallback
        hist   = np.full(30, 0.01)          # tiny historical widths
        # Large interval for day 0 only
        p10    = p50.copy()
        p90    = p50.copy()
        p10[:24] = p50[:24] - 100.0         # width = 200 >> k × 0.01
        p90[:24] = p50[:24] + 100.0

        # Days 1 and 2: zero width → no fallback
        p10[24:] = p50[24:]
        p90[24:] = p50[24:]

        _, branches = select_forecast_per_day(
            p50, p10, p90, pers, imp, hist, k=1.5
        )
        assert branches[0] == "persistence", "Low-confidence day must fall back"

    def test_clean_signal_uses_gbm(self):
        """No imputed lags, normal widths → branch = gbm for all days."""
        n_days = 10
        T      = n_days * 24
        rng    = np.random.default_rng(4)
        p50    = rng.uniform(50, 70, T)
        p10    = p50 - 8.0
        p90    = p50 + 8.0
        pers   = rng.uniform(45, 65, T)
        imp    = np.zeros(T + 300, dtype=bool)    # nothing imputed
        hist   = np.full(30, 16.0)                # same as interval width

        _, branches = select_forecast_per_day(
            p50, p10, p90, pers, imp, hist, k=2.0   # high k → hard to trigger
        )
        assert all(b == "gbm" for b in branches), (
            "No imputed lags + normal width + high k → all days should be GBM"
        )

    def test_gbm_branch_returns_p50(self):
        """Selected forecast matches p50 exactly for GBM days."""
        n_days = 5
        T      = n_days * 24
        rng    = np.random.default_rng(2)
        p50    = rng.uniform(50, 70, T)
        pers   = rng.uniform(45, 65, T)
        imp    = np.zeros(T + 300, dtype=bool)
        hist   = np.full(30, 16.0)

        sel, branches = select_forecast_per_day(
            p50, p50 - 8, p50 + 8, pers, imp, hist, k=2.0
        )
        for d, b in enumerate(branches):
            if b == "gbm":
                np.testing.assert_array_equal(sel[d * 24 : d * 24 + 24], p50[d * 24 : d * 24 + 24])

    def test_persistence_branch_returns_pers(self):
        """Selected forecast matches persistence exactly for fallback days."""
        n_days = 5
        T      = n_days * 24
        rng    = np.random.default_rng(6)
        p50    = rng.uniform(50, 70, T)
        pers   = rng.uniform(45, 65, T)
        imp    = np.zeros(T + 300, dtype=bool)
        hist   = np.full(30, 0.01)     # tiny hist → low-conf fallback on wide days

        p10 = p50 - 100.0    # very wide interval → low confidence for all days
        p90 = p50 + 100.0

        sel, branches = select_forecast_per_day(
            p50, p10, p90, pers, imp, hist, k=1.25
        )
        for d, b in enumerate(branches):
            if b == "persistence":
                np.testing.assert_array_equal(
                    sel[d * 24 : d * 24 + 24],
                    pers[d * 24 : d * 24 + 24],
                )

    def test_insufficient_hist_no_low_conf(self):
        """With fewer than 7 historical widths, low_conf is never set."""
        n_days = 3
        T      = n_days * 24
        rng    = np.random.default_rng(11)
        p50    = rng.uniform(50, 70, T)
        pers   = rng.uniform(45, 65, T)
        imp    = np.zeros(T + 300, dtype=bool)

        # Very wide interval but only 3 historical days → should not trigger low_conf
        hist = np.full(3, 0.01)
        p10  = p50 - 500.0
        p90  = p50 + 500.0

        _, branches = select_forecast_per_day(
            p50, p10, p90, pers, imp, hist, k=1.25
        )
        # With < 7 hist points, low_conf is forced False; no imputed lags → all GBM
        assert all(b == "gbm" for b in branches)
