"""Tests for all ValueGenerators (8 types) including state updates and side effects."""

import datetime as dt

import numpy as np

from financial_simulator.core.distributions import ConstantDistribution, NormalDistribution
from financial_simulator.core.event import (
    DistributionValue,
    DividendValue,
    FixedValue,
    GrowingValue,
    InvestmentContributionValue,
    RateChangeValue,
    TaxEventValue,
    VariableRateLoanValue,
)

# --- FixedValue ---


def test_fixed_value_pos_and_neg():
    gen = FixedValue(1234.5)
    val, meta = gen.get_value(dt.datetime(2026, 1, 1), {})
    assert val == 1234.5
    assert meta == {}

    gen2 = FixedValue(value=-999)
    assert gen2.get_value(dt.datetime(2026, 1, 1), {})[0] == -999


# --- GrowingValue ---


def test_growing_value_compounds_over_time():
    gen = GrowingValue(initial=1000.0, growth_rate=0.12)
    gen.reset()
    t1 = dt.datetime(2026, 1, 1)
    t2 = dt.datetime(2027, 1, 1)

    val1, _ = gen.get_value(t1, {})
    val2, _ = gen.get_value(t2, {})
    assert val2 > val1 * 1.11


def test_growing_value_zero_delta_no_change():
    gen = GrowingValue(initial=5000.0, growth_rate=0.10)
    gen.reset()
    t = dt.datetime(2026, 3, 15)
    val1, _ = gen.get_value(t, {})
    val2, _ = gen.get_value(t, {})  # same timestamp
    assert val1 == val2


# --- DistributionValue ---


def test_distribution_value_reproducible():
    dist = NormalDistribution(mean=100, std=5)
    gen = DistributionValue(dist=dist)

    rng = np.random.default_rng(99)
    val1 = gen.get_value(dt.datetime(2026, 1, 1), {}, rng)[0]

    rng = np.random.default_rng(99)
    val2 = gen.get_value(dt.datetime(2026, 1, 1), {}, rng)[0]

    assert val1 == val2


def test_distribution_value_uses_passed_rng():
    dist = ConstantDistribution(value=42.0)
    gen = DistributionValue(dist=dist)
    rng = np.random.default_rng(1)
    val, _ = gen.get_value(dt.datetime(2026, 1, 1), {}, rng)
    assert val == 42.0


# --- RateChangeValue (state_update) ---


def test_rate_change_value_emits_zero_cash_and_state_update():
    dist = NormalDistribution(mean=0.05, std=0.001)
    gen = RateChangeValue(dist=dist, update_key="interest_rate")
    gen.reset()
    rng = np.random.default_rng(7)
    val, meta = gen.get_value(dt.datetime(2026, 6, 1), {}, rng)
    assert val == 0.0
    assert "state_update" in meta
    assert "interest_rate" in meta["state_update"]
    assert isinstance(meta["state_update"]["interest_rate"], float)


# --- VariableRateLoanValue ---


def test_variable_rate_loan_basic_payment_and_metadata():
    gen = VariableRateLoanValue(
        principal=120000, initial_rate=0.06, term_months=360, rate_key="rate"
    )
    state = {"rate": 0.06}
    val, meta = gen.get_value(dt.datetime(2026, 1, 1), state)
    assert val < 0  # payment is outflow
    assert "interest" in meta
    assert "principal" in meta
    assert "remaining_balance" in meta
    assert meta["remaining_balance"] < 120000


def test_variable_rate_loan_respects_state_rate_changes():
    gen = VariableRateLoanValue(principal=100000, initial_rate=0.05, term_months=120, rate_key="r")
    state = {"r": 0.05}
    gen.get_value(dt.datetime(2026, 1, 1), state)  # month 1
    state["r"] = 0.08  # rate spike
    val2, meta2 = gen.get_value(dt.datetime(2026, 2, 1), state)
    assert meta2["rate"] == 0.08


def test_variable_rate_loan_zero_rate_path():
    gen = VariableRateLoanValue(principal=12000, initial_rate=0.0, term_months=12, rate_key="r")
    state = {"r": 0.0}
    val, meta = gen.get_value(dt.datetime(2026, 1, 1), state)
    assert val < 0
    assert meta["remaining_balance"] == 11000.0  # 12000 - 1000


def test_variable_rate_loan_exhausts_after_term():
    gen = VariableRateLoanValue(principal=1000, initial_rate=0.0, term_months=2, rate_key="r")
    state = {"r": 0.0}
    gen.get_value(dt.datetime(2026, 1, 1), state)
    gen.get_value(dt.datetime(2026, 2, 1), state)
    val3, _ = gen.get_value(dt.datetime(2026, 3, 1), state)
    assert val3 == 0.0


# --- DividendValue ---


def test_dividend_value_uses_portfolio_key():
    gen = DividendValue(annual_yield=0.04, investment_value_key="portfolio")
    state = {"portfolio": 120000.0}
    val, meta = gen.get_value(dt.datetime(2026, 1, 1), state)
    assert abs(val - 120000 * 0.04 / 12) < 0.01
    assert meta.get("source") == "dividend"


def test_dividend_value_missing_key_defaults_to_zero():
    gen = DividendValue(annual_yield=0.05)
    val, _ = gen.get_value(dt.datetime(2026, 1, 1), {})
    assert val == 0.0


# --- InvestmentContributionValue ---


def test_investment_contribution_simple():
    gen = InvestmentContributionValue(amount=1500.0)
    val, meta = gen.get_value(dt.datetime(2026, 1, 1), {})
    assert val == -1500.0
    assert meta.get("type") == "investment_contribution"


def test_investment_contribution_with_growth_key():
    gen = InvestmentContributionValue(amount=2000.0, growth_key="inflation_factor")
    state = {"inflation_factor": 1.03}
    val, _ = gen.get_value(dt.datetime(2026, 1, 1), state)
    assert val == -2060.0


# --- TaxEventValue (state_update) ---


def test_tax_event_value_applies_rate_and_accumulates():
    gen = TaxEventValue(rate=0.22, base_key="taxable_income", tax_key="taxes")
    state = {"taxable_income": 10000.0, "taxes": 500.0}
    val, meta = gen.get_value(dt.datetime(2026, 4, 1), state)
    assert val == -2200.0
    assert meta["tax"] == 2200.0
    assert meta["state_update"]["taxes"] == 2700.0


def test_tax_event_value_missing_base_defaults_to_zero():
    gen = TaxEventValue(rate=0.15, base_key="income")
    val, meta = gen.get_value(dt.datetime(2026, 1, 1), {})
    assert val == 0.0
    assert meta["tax"] == 0.0
