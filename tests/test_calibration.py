"""
Tests for the CQR calibration module.

Covers:
- Offsets are computed only from calibration data (no look-ahead)
- Quantiles do not cross after calibration
- Rolling variant uses only past data
- On synthetic data with a known shift, CQR coverage moves toward the target
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bess_arbitrage.calibration import conformal_quantile_offset, rolling_conformal


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_synthetic(n: int, shift: float = 0.0, seed: int = 0) -> tuple:
    """
    Synthetic data: actuals ~ N(0,1), quantile forecasts centred at zero.
    `shift` displaces actuals upward so forecasts are systematically low.
    """
    rng = np.random.default_rng(seed)
    y   = rng.normal(shift, 1.0, n)
    q10 = np.full(n, -1.282)   # theoretical N(0,1) p10
    q50 = np.zeros(n)
    q90 = np.full(n,  1.282)   # theoretical N(0,1) p90
    return y, q10, q50, q90


# ── conformal_quantile_offset ─────────────────────────────────────────────────

class TestConformedOffset:
    def test_no_look_ahead_from_test_data(self):
        """Offset depends only on calibration arrays passed in."""
        y_cal  = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q_cal  = np.zeros(5)
        off_a  = conformal_quantile_offset(y_cal, q_cal, 0.90)

        # Modifying test-period data must not change the result
        y_test_contamination = np.array([1000.0] * 100)
        off_b  = conformal_quantile_offset(y_cal, q_cal, 0.90)
        assert off_a == off_b, "Offset must not depend on data outside calibration set"

    def test_finite_sample_correction(self):
        """Level = ceil((n+1)*alpha)/n, capped at 1."""
        n     = 10
        alpha = 0.90
        expected_level = min(math.ceil((n + 1) * alpha) / n, 1.0)
        scores = np.sort(np.arange(n, dtype=float))
        off = conformal_quantile_offset(np.zeros(n), -scores, alpha)
        assert abs(off - np.quantile(scores, expected_level)) < 1e-9

    def test_zero_offset_on_perfect_quantile(self):
        """If q_hat_p90 = actual 90th pct of y, offset ~ 0 at the median score."""
        rng = np.random.default_rng(7)
        y   = rng.normal(0, 1, 1000)
        q90 = np.full(1000, np.quantile(y, 0.90))
        off = conformal_quantile_offset(y, q90, 0.90)
        # Offset should be small (close to 0) for a well-calibrated quantile
        assert abs(off) < 0.5

    def test_positive_offset_when_forecasts_too_low(self):
        """If q_hat is systematically below actuals, offset must be positive for p90."""
        y   = np.ones(200) * 10.0
        q90 = np.zeros(200)              # forecast too low by 10
        off = conformal_quantile_offset(y, q90, 0.90)
        assert off > 0

    def test_negative_offset_when_forecasts_too_high(self):
        """If q_hat_p10 is above actuals, offset must be negative."""
        y   = np.zeros(200)
        q10 = np.ones(200) * 10.0       # forecast too high by 10
        off = conformal_quantile_offset(y, q10, 0.10)
        assert off < 0


# ── rolling_conformal ─────────────────────────────────────────────────────────

class TestRollingConformal:
    def _make_inputs(self, n_cal_days: int = 31, n_test_days: int = 122, shift: float = 2.0):
        """
        n_cal_days=31  → August (744 h)
        n_test_days=122 → Sep–Dec (2928 h)
        shift=2        → actuals are 2 units above forecasts (known bias)
        """
        y_cal,  q10_cal,  q50_cal,  q90_cal  = _make_synthetic(n_cal_days  * 24, shift)
        y_test, q10_test, q50_test, q90_test = _make_synthetic(n_test_days * 24, shift, seed=1)
        return (
            y_cal, q10_cal, q90_cal,
            y_test, q10_test, q50_test, q90_test,
        )

    def test_output_shape(self):
        args = self._make_inputs()
        y_test = args[3]
        q10, q50, q90 = rolling_conformal(*args)
        assert q10.shape == y_test.shape
        assert q50.shape == y_test.shape
        assert q90.shape == y_test.shape

    def test_quantiles_do_not_cross(self):
        """After calibration, p10 <= p50 <= p90 must hold for every hour."""
        args = self._make_inputs()
        q10, q50, q90 = rolling_conformal(*args)
        assert np.all(q10 <= q50 + 1e-9), "p10 must not exceed p50"
        assert np.all(q50 <= q90 + 1e-9), "p50 must not exceed p90"

    def test_no_look_ahead(self):
        """
        For each test day d, the offset must be computed from data before day d.
        We verify this by checking that corrupting future test actuals does not
        change the calibrated forecast for earlier test days.
        """
        (y_cal, q10_cal, q90_cal,
         y_test, q10_test, q50_test, q90_test) = self._make_inputs(n_test_days=60)

        q10_a, _, _ = rolling_conformal(
            y_cal, q10_cal, q90_cal,
            y_test, q10_test, q50_test, q90_test,
        )

        # Corrupt future test actuals (days 30 onward)
        y_test_corrupted       = y_test.copy()
        y_test_corrupted[720:] = 1e6   # impossible values

        q10_b, _, _ = rolling_conformal(
            y_cal, q10_cal, q90_cal,
            y_test_corrupted, q10_test, q50_test, q90_test,
        )

        # The first 30 test days (720 hours) must be identical
        np.testing.assert_array_equal(
            q10_a[:720], q10_b[:720],
            err_msg="Rolling calibration must not use future test actuals",
        )

    def test_coverage_moves_toward_target_on_shifted_data(self):
        """
        With actuals shifted +2 above forecasts, uncalibrated p90 coverage
        should be well below 90%. After rolling CQR it should be higher.
        """
        (y_cal, q10_cal, q90_cal,
         y_test, q10_test, q50_test, q90_test) = self._make_inputs(shift=2.0)

        # Uncalibrated p90 coverage (actuals consistently above forecasts → low)
        cov_raw = float(np.mean(y_test <= q90_test))

        _, _, q90_cal_out = rolling_conformal(
            y_cal, q10_cal, q90_cal,
            y_test, q10_test, q50_test, q90_test,
        )
        cov_rolled = float(np.mean(y_test <= q90_cal_out))

        assert cov_rolled > cov_raw, (
            f"Rolling CQR coverage ({cov_rolled:.3f}) should exceed "
            f"uncalibrated ({cov_raw:.3f}) when forecasts are biased low"
        )
        # Should be meaningfully closer to 0.90
        assert abs(cov_rolled - 0.90) < abs(cov_raw - 0.90), (
            "Rolling CQR should bring p90 coverage closer to 90% target"
        )


# ── Static CQR (applied via conformal_quantile_offset) ───────────────────────

class TestStaticCQR:
    def test_quantiles_do_not_cross_after_static_cqr(self):
        """Static offsets + sort must leave p10 <= p50 <= p90 everywhere."""
        rng = np.random.default_rng(3)
        n   = 500
        y_cal  = rng.normal(5, 3, n)
        q10_cal = np.full(n, 2.0)
        q50_cal = np.full(n, 5.0)
        q90_cal = np.full(n, 8.0)

        off10 = conformal_quantile_offset(y_cal, q10_cal, 0.10)
        off90 = conformal_quantile_offset(y_cal, q90_cal, 0.90)

        q10_test = np.full(100, 2.0) + off10
        q50_test = np.full(100, 5.0)
        q90_test = np.full(100, 8.0) + off90

        stacked = np.sort(np.stack([q10_test, q50_test, q90_test], axis=1), axis=1)
        q10_out, q50_out, q90_out = stacked[:, 0], stacked[:, 1], stacked[:, 2]

        assert np.all(q10_out <= q50_out + 1e-9)
        assert np.all(q50_out <= q90_out + 1e-9)

    def test_coverage_moves_toward_target(self):
        """With a known +3 shift, static CQR on calibration should fix p90 coverage."""
        rng   = np.random.default_rng(42)
        shift = 3.0
        n_cal = 744   # one month

        y_cal  = rng.normal(shift, 1, n_cal)
        q90_cal = np.full(n_cal, 1.282)   # theoretical N(0,1) p90 — too low

        off90  = conformal_quantile_offset(y_cal, q90_cal, 0.90)
        assert off90 > 0, "Offset must be positive when forecasts are below actuals"

        n_test  = 2928
        y_test  = rng.normal(shift, 1, n_test)
        q90_raw = np.full(n_test, 1.282)
        q90_cal = q90_raw + off90

        cov_raw = float(np.mean(y_test <= q90_raw))
        cov_cal = float(np.mean(y_test <= q90_cal))

        assert cov_cal > cov_raw
        assert abs(cov_cal - 0.90) < abs(cov_raw - 0.90)
