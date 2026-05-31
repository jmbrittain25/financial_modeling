"""Tests for value generators."""

import pytest
from datetime import datetime, timedelta
import numpy as np

from financial_simulator.core.event import (
    FixedValue,
    GrowingValue,
    DistributionValue,
    VariableRateLoanValue,
)
from financial_simulator.core.distributions import NormalDistribution


def test_fixed_value():
    gen = FixedValue(value=500.0)
    val, meta = gen.get_value(datetime(2026, 1, 1), {})
    assert val == 500.0
    assert meta == {}


def test_growing_value():
    gen = GrowingValue(initial=1000.0, growth_rate=0.12)
    gen.reset()
    t1 = datetime(2026, 1, 1)
    t2 = datetime(2027, 1, 1)

    val1, _ = gen.get_value(t1, {})
    val2, _ = gen.get_value(t2, {})
    assert val2 > val1 * 1.11  # should have grown


def test_distribution_value_reproducible():
    dist = NormalDistribution(mean=100, std=5)
    gen = DistributionValue(dist=dist)

    rng = np.random.default_rng(99)
    val1 = gen.get_value(datetime(2026, 1, 1), {}, rng)[0]

    rng = np.random.default_rng(99)
    val2 = gen.get_value(datetime(2026, 1, 1), {}, rng)[0]

    assert val1 == val2


def test_variable_rate_loan_smoke():
    """Basic smoke test that the loan generator can be instantiated and called."""
    gen = VariableRateLoanValue(
        principal=120000,
        initial_rate=0.06,
        term_months=360,
        rate_key="rate"
    )
    state = {"rate": 0.06}
    val, meta = gen.get_value(datetime(2026, 1, 1), state)
    assert val <= 0  # should produce a payment (negative or zero)
