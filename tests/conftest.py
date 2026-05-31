"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np


@pytest.fixture
def seeded_rng():
    """Provide a reproducible random number generator."""
    return np.random.default_rng(42)
