"""Tests for stochastic processes (GeometricBrownianMotion, MeanRevertingProcess).

Includes edge cases for delta <= 0 and reproducibility.
"""

import datetime as dt

import numpy as np

from financial_simulator.core.stochastic import (
    GeometricBrownianMotion,
    MeanRevertingProcess,
)

# --- GBM ---


def test_gbm_positive_drift_increases_on_average():
    gbm = GeometricBrownianMotion(drift=0.20, volatility=0.10)
    rng = np.random.default_rng(42)
    vals = [100.0]
    for _ in range(24):
        vals.append(gbm.step(vals[-1], dt.timedelta(days=30), rng))
    # With 20% drift, after 2 years we expect significant growth on average
    assert vals[-1] > 100.0 * 1.2


def test_gbm_reproducible_with_same_rng_state():
    gbm = GeometricBrownianMotion(drift=0.08, volatility=0.25)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    v1 = gbm.step(100.0, dt.timedelta(days=365), rng1)
    v2 = gbm.step(100.0, dt.timedelta(days=365), rng2)
    assert v1 == v2


def test_gbm_zero_or_negative_delta_returns_current():
    gbm = GeometricBrownianMotion(drift=0.10, volatility=0.20)
    rng = np.random.default_rng(1)
    assert gbm.step(123.0, dt.timedelta(0), rng) == 123.0
    assert gbm.step(123.0, dt.timedelta(days=-5), rng) == 123.0


# --- Mean Reverting ---


def test_mean_reverting_moves_toward_long_term_mean():
    mr = MeanRevertingProcess(long_term_mean=0.04, speed=3.0, volatility=0.005)
    rng = np.random.default_rng(77)
    val = 0.12  # start far from mean
    for _ in range(12):
        val = mr.step(val, dt.timedelta(days=30), rng)
    # After a year with high speed, should be much closer to 4%
    assert abs(val - 0.04) < 0.03


def test_mean_reverting_reproducible():
    mr = MeanRevertingProcess(long_term_mean=50.0, speed=1.0, volatility=2.0)
    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)
    v1 = mr.step(60.0, dt.timedelta(days=90), rng1)
    v2 = mr.step(60.0, dt.timedelta(days=90), rng2)
    assert v1 == v2


def test_mean_reverting_zero_delta_no_change():
    mr = MeanRevertingProcess(long_term_mean=10.0, speed=2.0, volatility=1.0)
    rng = np.random.default_rng(3)
    assert mr.step(7.5, dt.timedelta(0), rng) == 7.5
