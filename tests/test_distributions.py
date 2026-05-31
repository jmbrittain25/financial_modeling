"""Tests for core distributions."""

import pytest
import numpy as np

from financial_simulator.core.distributions import (
    NormalDistribution,
    UniformDistribution,
    TriangularDistribution,
    BetaDistribution,
    LogNormalDistribution,
    create_distribution,
)


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


def test_beta_distribution():
    dist = BetaDistribution(alpha=2, beta=5)
    rng = np.random.default_rng(42)
    samples = [dist.sample(rng) for _ in range(1000)]
    assert all(0 <= s <= 1 for s in samples)
    # Mean of Beta(2,5) is 2/7 ≈ 0.2857
    assert abs(np.mean(samples) - 0.2857) < 0.02


def test_create_distribution():
    data = {"type": "triangular", "low": 1, "mode": 2, "high": 3}
    dist = create_distribution(data)
    assert isinstance(dist, TriangularDistribution)
    assert dist.low == 1


def test_reproducibility_with_rng():
    dist = NormalDistribution(mean=0, std=1)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    assert dist.sample(rng1) == dist.sample(rng2)
