"""Clean Pydantic models for common financial objects.

These models are designed to be used both for configuration and as
rich objects inside simulations (loans, tax schedules, portfolios, etc.).
They are intentionally kept simple and extensible.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Loan(BaseModel):
    """Represents a loan with amortization characteristics."""

    model_config = ConfigDict(extra="forbid")

    name: str
    principal: float
    annual_rate: float
    term_months: int
    payment_frequency: int = 12  # payments per year

    def monthly_payment(self) -> float:
        """Calculate the fixed monthly payment (standard amortization)."""
        if self.annual_rate == 0:
            return self.principal / self.term_months

        monthly_rate = self.annual_rate / 12
        r = (1 + monthly_rate) ** self.term_months
        return self.principal * monthly_rate * r / (r - 1)


class TaxBracket(BaseModel):
    """Single tax bracket."""

    model_config = ConfigDict(extra="forbid")

    lower: float
    upper: float | None = None
    rate: float


class TaxSchedule(BaseModel):
    """Progressive tax schedule (e.g. federal income tax brackets)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    brackets: list[TaxBracket]
    standard_deduction: float = 0.0

    def effective_rate(self, taxable_income: float) -> float:
        """Compute the effective (average) tax rate for a given income."""
        if taxable_income <= self.standard_deduction:
            return 0.0

        taxable = taxable_income - self.standard_deduction
        tax = 0.0
        prev = 0.0

        for bracket in self.brackets:
            upper = bracket.upper if bracket.upper is not None else taxable
            segment = min(taxable, upper) - prev
            if segment > 0:
                tax += segment * bracket.rate
            prev = upper if bracket.upper is not None else taxable
            if taxable <= upper:
                break

        return tax / taxable_income if taxable_income > 0 else 0.0


class Portfolio(BaseModel):
    """Simple portfolio definition."""

    model_config = ConfigDict(extra="forbid")

    name: str
    assets: dict[str, float] = Field(default_factory=dict)  # asset_name -> allocation %

    def normalize(self) -> Portfolio:
        """Return a new portfolio with allocations normalized to sum to 1."""
        total = sum(self.assets.values())
        if total == 0:
            return self
        return Portfolio(
            name=self.name,
            assets={k: v / total for k, v in self.assets.items()},
        )
