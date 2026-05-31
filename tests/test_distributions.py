"""Tests for core distributions (all 7 types + factory + validation + reproducibility)."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import floats

from financial_simulator.core.distributions import (
    NormalDistribution,
    UniformDistribution,
    TriangularDistribution,
    LogNormalDistribution,
    ExponentialDistribution,
    ConstantDistribution,
    BetaDistribution,
    create_distribution,
    AnyDistribution,
)


# --- Basic sampling and statistical sanity ---

def test_normal_distribution():
    dist = NormalDistribution(mean=100, std=10)
    rng = np.random.default_rng(42)
    samples = [dist.sample(rng) for _ in range(1000)]
    assert abs(np.mean(samples) - 100) < 1.0
    assert abs(np.std(samples) - 10) < 1.5


def test_uniform_distribution():
    dist = UniformDistribution(low=10, high=20)
    rng = np.random.default_rng(42)
    samples = [dist.sample(rng) for _ in range(500)]
    assert min(samples) >= 10
    assert max(samples) <= 20
    assert abs(np.mean(samples) - 15) < 0.5


def test_triangular_distribution():
    dist = TriangularDistribution(low=0, mode=5, high=10)
    rng = np.random.default_rng(42)
    samples = [dist.sample(rng) for _ in range(2000)]
    assert min(samples) >= 0
    assert max(samples) <= 10
    # Mode should be most frequent region (rough check)
    hist, _ = np.histogram(samples, bins=10, range=(0, 10))
    # The bin containing the mode (around 5) should have high count
    assert hist[4] + hist[5] >= max(hist)  # loose


def test_lognormal_distribution():
    dist = LogNormalDistribution(mean=0.0, sigma=0.5)
    rng = np.random.default_rng(42)
    samples = [dist.sample(rng) for _ in range(2000)]
    assert all(s > 0 for s in samples)
    # Median of lognormal(mean, sigma) is exp(mean)
    assert abs(np.median(samples) - np.exp(0.0)) < 0.1


def test_exponential_distribution():
    rate = 2.0
    dist = ExponentialDistribution(rate=rate)
    rng = np.random.default_rng(42)
    samples = [dist.sample(rng) for _ in range(3000)]
    assert all(s > 0 for s in samples)
    # Mean of exponential(rate) = 1/rate
    assert abs(np.mean(samples) - 1.0 / rate) < 0.05


def test_constant_distribution():
    dist = ConstantDistribution(value=42.0)
    rng = np.random.default_rng(999)
    for _ in range(100):
        assert dist.sample(rng) == 42.0


def test_beta_distribution():
    dist = BetaDistribution(alpha=2, beta=5)
    rng = np.random.default_rng(42)
    samples = [dist.sample(rng) for _ in range(1000)]
    assert all(0 <= s <= 1 for s in samples)
    # Mean of Beta(2,5) is 2/7 ≈ 0.2857
    assert abs(np.mean(samples) - 0.2857) < 0.02


# --- Validation errors ---

@pytest.mark.parametrize("low,high", [(5, 1), (0, -1), (10, 10 - 1e-12)])
def test_uniform_validation_rejects_low_gt_high(low, high):
    with pytest.raises(Exception):  # Pydantic ValidationError
        UniformDistribution(low=low, high=high)


@pytest.mark.parametrize("low,mode,high", [
    (5, 3, 10),
    (0, 10, 5),
    (1, 1, 0),
])
def test_triangular_validation_rejects_invalid(low, mode, high):
    with pytest.raises(Exception):
        TriangularDistribution(low=low, mode=mode, high=high)


@pytest.mark.parametrize("rate", [0, -0.1, -1])
def test_exponential_validation_rejects_non_positive_rate(rate):
    with pytest.raises(Exception):
        ExponentialDistribution(rate=rate)


@pytest.mark.parametrize("alpha,beta", [(0, 1), (1, 0), (-0.5, 2), (2, -1)])
def test_beta_validation_rejects_non_positive_params(alpha, beta):
    with pytest.raises(Exception):
        BetaDistribution(alpha=alpha, beta=beta)


# --- create_distribution factory ---

def test_create_distribution_from_dict_triangular():
    data = {"type": "triangular", "low": 1, "mode": 2, "high": 3}
    dist = create_distribution(data)
    assert isinstance(dist, TriangularDistribution)
    assert dist.low == 1 and dist.mode == 2 and dist.high == 3


def test_create_distribution_from_dict_normal():
    dist = create_distribution({"type": "normal", "mean": 0, "std": 1})
    assert isinstance(dist, NormalDistribution)


def test_create_distribution_passthrough():
    original = NormalDistribution(mean=5, std=2)
    result = create_distribution(original)
    assert result is original


def test_create_distribution_rejects_bad_type():
    with pytest.raises(TypeError):
        create_distribution(123)


def test_create_distribution_rejects_unknown_type():
    with pytest.raises(Exception):
        create_distribution({"type": "weird", "foo": 1})


# --- Reproducibility across all distribution types ---

@pytest.mark.parametrize("dist_cls,kwargs", [
    (NormalDistribution, {"mean": 0, "std": 1}),
    (UniformDistribution, {"low": -10, "high": 10}),
    (TriangularDistribution, {"low": 0, "mode": 5, "high": 10}),
    (LogNormalDistribution, {"mean": 1.2, "sigma": 0.3}),
    (ExponentialDistribution, {"rate": 0.5}),
    (ConstantDistribution, {"value": -999.0}),
    (BetaDistribution, {"alpha": 1.5, "beta": 3.2}),
])
def test_all_distributions_are_reproducible(dist_cls, kwargs):
    dist = dist_cls(**kwargs)
    rng1 = np.random.default_rng(777)
    rng2 = np.random.default_rng(777)
    v1 = dist.sample(rng1)
    v2 = dist.sample(rng2)
    assert v1 == v2, f"{dist_cls.__name__} not reproducible"


# --- Hypothesis property-based tests ---

@settings(max_examples=200)
@given(
    mean=floats(-1e6, 1e6),
    std=floats(1e-6, 1e6),
    n=st.integers(10, 500),
)
def test_normal_samples_within_reasonable_bounds(mean, std, n):
    dist = NormalDistribution(mean=mean, std=std)
    rng = np.random.default_rng(12345)
    samples = [dist.sample(rng) for _ in range(n)]
    # Almost all samples should be within mean ± 6*std (very conservative)
    lo, hi = mean - 6 * std, mean + 6 * std
    assert all(lo <= s <= hi for s in samples)


@settings(max_examples=100)
@given(low=floats(-1e5, 1e5), width=floats(1e-9, 1e5))
def test_uniform_respects_bounds(low, width):
    high = low + width
    dist = UniformDistribution(low=low, high=high)
    rng = np.random.default_rng(42)
    samples = [dist.sample(rng) for _ in range(50)]
    assert all(low <= s <= high for s in samples)


@settings(max_examples=50)
@given(alpha=floats(0.01, 50), beta=floats(0.01, 50))
def test_beta_always_in_unit_interval(alpha, beta):
    dist = BetaDistribution(alpha=alpha, beta=beta)
    rng = np.random.default_rng(99)
    samples = [dist.sample(rng) for _ in range(30)]
    assert all(0.0 <= s <= 1.0 for s in samples)
