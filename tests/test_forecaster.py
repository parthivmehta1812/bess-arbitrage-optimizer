"""
Tests for the forecaster module.

Covers:
- _make_features produces no look-ahead: same-day prices (day_start .. t)
  must not affect any feature.
- _make_features IS sensitive to previous-day prices.
- forecast_prices returns p10 <= p50 <= p90 for every hour.
"""

from __future__ import annotations

import numpy as np
import pytest

from bess_arbitrage.forecaster import _make_features, train_price_forecaster, forecast_prices


# ── helpers ───────────────────────────────────────────────────────────────────

def _base_prices(n: int = 8760, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(10.0, 200.0, n).astype(float)


# ── no look-ahead ─────────────────────────────────────────────────────────────

class TestNoLookAhead:
    """
    For a forecast of hour *t*, the bid is placed at noon on day D-1.
    At that time prices[day_start .. t] (same day, up to and including t)
    are not yet known.  Changing any of those values must leave
    _make_features(prices, t) unchanged.
    """

    @pytest.mark.parametrize("t", [24, 25, 36, 47, 168, 500, 1001])
    def test_same_day_prices_do_not_affect_features(self, t: int):
        prices = _base_prices()
        expected = _make_features(prices, t)

        hod = t % 24
        day_start = t - hod  # midnight of current day D

        # Corrupt every price in [day_start, t] (same day, up to and including t)
        perturbed = prices.copy()
        perturbed[day_start : t + 1] = 9999.0

        result = _make_features(perturbed, t)

        assert result == expected, (
            f"Hour t={t}: changing prices[{day_start}:{t+1}] "
            f"(same-day, up to t) must not change any feature.\n"
            f"Before: {expected}\nAfter:  {result}"
        )

    @pytest.mark.parametrize("t", [48, 72, 168, 500, 1001])
    def test_previous_day_prices_do_affect_features(self, t: int):
        """Changing a price on day D-1 must change at least one feature."""
        prices = _base_prices()
        hod = t % 24
        day_start = t - hod

        # Change a price inside the previous-day rolling window
        # [day_start-24 .. day_start) — this is the window used by roll_mean/std
        perturbed = prices.copy()
        prev_day_idx = day_start - 1   # last hour of day D-1
        if prev_day_idx >= 0:
            perturbed[prev_day_idx] = prices[prev_day_idx] + 1000.0

        original = _make_features(prices, t)
        changed  = _make_features(perturbed, t)

        assert original != changed, (
            f"Hour t={t}: changing prices[{prev_day_idx}] (day D-1) "
            f"must change at least one feature."
        )

    def test_hour_zero_of_day_does_not_crash(self):
        """Hour 0 of a day (hod=0) must not crash; day_start == t."""
        prices = _base_prices()
        for t in [0, 24, 48, 168]:
            feat = _make_features(prices, t)
            assert len(feat) == 11

    def test_early_hours_use_zeros_safely(self):
        """For t < 24 there is no full previous day; roll_mean/std fall back to 0."""
        prices = _base_prices()
        for t in range(24):
            feat = _make_features(prices, t)
            # roll_mean and roll_std are the last two features
            assert feat[-2] == 0.0, f"t={t}: roll_mean should be 0 (no prior day)"
            assert feat[-1] == 0.0, f"t={t}: roll_std should be 0 (no prior day)"


# ── quantile ordering ─────────────────────────────────────────────────────────

class TestQuantileOrdering:
    """
    forecast_prices must return p10 <= p50 <= p90 for every hour,
    even though the three models are trained independently.
    """

    def test_no_quantile_crossing_on_real_data(self):
        prices = _base_prices(n=8760)

        # Minimal training: hours 168–1416 (Jan–Feb), predict Mar (1416–2160)
        train_end = 1416
        test_start = 1416
        test_end   = 2160

        models = train_price_forecaster(prices, train_end_h=train_end, n_estimators=50)
        p50, p10, p90 = forecast_prices(models, prices, test_start, test_end)

        assert np.all(p10 <= p50 + 1e-9), "p10 must not exceed p50 anywhere"
        assert np.all(p50 <= p90 + 1e-9), "p50 must not exceed p90 anywhere"

    def test_quantile_ordering_with_adversarial_prices(self):
        """
        On a price series with wild swings, ordering must still hold.
        """
        rng = np.random.default_rng(99)
        prices = rng.uniform(-100, 600, 8760).astype(float)

        train_end  = 1416
        test_start = 1416
        test_end   = 1440   # 24 hours only (fast)

        models = train_price_forecaster(prices, train_end_h=train_end, n_estimators=30)
        p50, p10, p90 = forecast_prices(models, prices, test_start, test_end)

        assert np.all(p10 <= p50 + 1e-9)
        assert np.all(p50 <= p90 + 1e-9)
