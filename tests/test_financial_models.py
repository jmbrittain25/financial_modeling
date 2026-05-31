"""Tests for clean financial domain models (Loan, TaxSchedule, Portfolio)."""

import pytest

from financial_simulator.core.financial_models import Loan, TaxBracket, TaxSchedule, Portfolio


# --- Loan ---

def test_loan_standard_amortization():
    loan = Loan(name="Mortgage", principal=300000, annual_rate=0.065, term_months=360)
    pmt = loan.monthly_payment()
    assert 1800 < pmt < 2000


def test_loan_zero_rate_divides_principal():
    loan = Loan(name="Zero", principal=12000, annual_rate=0.0, term_months=12)
    assert loan.monthly_payment() == 1000.0


# --- TaxSchedule ---

def test_tax_schedule_effective_rate_progressive():
    brackets = [
        TaxBracket(lower=0, upper=10000, rate=0.10),
        TaxBracket(lower=10000, upper=40000, rate=0.20),
        TaxBracket(lower=40000, upper=None, rate=0.30),
    ]
    ts = TaxSchedule(name="Fed", brackets=brackets, standard_deduction=1000)
    eff = ts.effective_rate(51000)
    # Rough manual: taxable=50k, tax ~ (9k*0.1 + 30k*0.2 + 10k*0.3) / 51k
    assert 0.18 < eff < 0.22


def test_tax_schedule_below_deduction_zero():
    ts = TaxSchedule(name="T", brackets=[TaxBracket(lower=0, upper=None, rate=0.25)], standard_deduction=5000)
    assert ts.effective_rate(3000) == 0.0


# --- Portfolio ---

def test_portfolio_normalize():
    p = Portfolio(name="Mixed", assets={"a": 60, "b": 30, "c": 10})
    norm = p.normalize()
    assert abs(sum(norm.assets.values()) - 1.0) < 1e-12
    assert norm.assets["a"] == 0.6


def test_portfolio_normalize_zero_total_is_noop():
    p = Portfolio(name="Zero", assets={})
    assert p.normalize() is p  # or at least returns equivalent with same name