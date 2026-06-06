"""
Value Generator Editor — Dynamic form for AnyValueGenerator.

Supports Fixed, Growing, Distribution, VariableRateLoan, Dividend, InvestmentContribution,
TaxEvent, and RateChange. When a generator needs a distribution it embeds the live
render_distribution_picker (no save section).
"""

from __future__ import annotations

from app.components.distribution_viz import render_distribution_picker
from financial_simulator.core.distributions import NormalDistribution
from financial_simulator.core.event import (
    AnyValueGenerator,
    DistributionValue,
    DividendValue,
    FixedValue,
    GrowingValue,
    InvestmentContributionValue,
    RateChangeValue,
    TaxEventValue,
    VariableRateLoanValue,
)

VG_TYPE_LABELS: dict[str, str] = {
    "Fixed": "Fixed amount (every time it fires)",
    "Growing": "Growing amount (fixed + annual growth %)",
    "Distribution": "Stochastic (draw from a distribution each time)",
    "VariableRateLoan": "Amortizing loan payment (rate comes from state key)",
    "Dividend": "Dividend / distribution income from an investment state var",
    "InvestmentContribution": "Regular contribution into an investment (negative cash)",
    "TaxEvent": "Tax computed as rate × base_state_key (writes to tax_key)",
    "RateChange": "Sample a distribution and write it into a state key (used by drivers)",
}

VG_TYPE_ORDER = list(VG_TYPE_LABELS.keys())


def get_default_vg(vg_type: str) -> AnyValueGenerator:
    if vg_type == "Fixed":
        return FixedValue(value=1000.0)
    if vg_type == "Growing":
        return GrowingValue(initial=1200.0, growth_rate=0.03)
    if vg_type == "Distribution":
        return DistributionValue(dist=NormalDistribution(mean=950.0, std=120.0))
    if vg_type == "VariableRateLoan":
        return VariableRateLoanValue(
            principal=250000.0, initial_rate=0.065, term_months=180, rate_key="mortgage_rate"
        )
    if vg_type == "Dividend":
        return DividendValue(annual_yield=0.028, investment_value_key="portfolio_value")
    if vg_type == "InvestmentContribution":
        return InvestmentContributionValue(amount=500.0, growth_key=None)
    if vg_type == "TaxEvent":
        return TaxEventValue(rate=0.22, base_key="taxable_income", tax_key="tax_paid")
    if vg_type == "RateChange":
        return RateChangeValue(
            dist=NormalDistribution(mean=0.065, std=0.008), update_key="market_rate"
        )
    return FixedValue(value=0.0)


def render_value_generator_editor(
    key_prefix: str = "vg",
    initial: AnyValueGenerator | None = None,
) -> AnyValueGenerator:
    """
    Live editor for value generators. Returns a ready-to-use instance.
    """
    import streamlit as st

    st.markdown("#### 💵 Value / Cash Flow")
    st.caption(
        "Enter positive amounts only. Use **Add to cash** / **Subtract from cash** on the "
        "generator to control direction."
    )

    current_type = getattr(initial, "type", "Fixed") if initial else "Fixed"

    label_to_key = {v: k for k, v in VG_TYPE_LABELS.items()}
    sel_label = st.selectbox(
        "Value Generator Type",
        options=list(VG_TYPE_LABELS.values()),
        index=VG_TYPE_ORDER.index(current_type) if current_type in VG_TYPE_ORDER else 0,
        key=f"{key_prefix}_vg_type",
        help="Pick the financial behavior. Many embed a live distribution picker below.",
    )
    vg_type = label_to_key[sel_label]

    # Seed from initial when type matches
    if initial is not None and getattr(initial, "type", None) == vg_type:
        # We'll read attributes below in the branches
        pass

    # --- Type-specific forms ---
    if vg_type == "Fixed":
        val = (
            float(getattr(initial, "value", 1000.0))
            if (initial and getattr(initial, "type", None) == "Fixed")
            else 1000.0
        )
        v = st.number_input(
            "Fixed amount each period",
            value=max(0.0, val),
            min_value=0.0,
            step=50.0,
            key=f"{key_prefix}_fixed_val",
        )
        return FixedValue(value=float(v))

    if vg_type == "Growing":
        init = (
            float(getattr(initial, "initial", 1000.0))
            if (initial and getattr(initial, "type", None) == "Growing")
            else 1000.0
        )
        g = (
            float(getattr(initial, "growth_rate", 0.03))
            if (initial and getattr(initial, "type", None) == "Growing")
            else 0.03
        )
        c1, c2 = st.columns(2)
        iv = c1.number_input(
            "Starting amount",
            value=max(0.0, init),
            min_value=0.0,
            step=50.0,
            key=f"{key_prefix}_grow_init",
        )
        gr = c2.number_input(
            "Annual growth rate (e.g. 0.03 = 3%)",
            value=g,
            step=0.005,
            format="%.3f",
            key=f"{key_prefix}_grow_rate",
        )
        return GrowingValue(initial=float(iv), growth_rate=float(gr))

    if vg_type == "Distribution":
        # Embed the excellent live distribution picker
        dist = NormalDistribution(mean=0.0, std=1.0)
        if (
            initial
            and getattr(initial, "type", None) == "Distribution"
            and hasattr(initial, "dist")
        ):
            dist = initial.dist  # type: ignore[attr-defined]
        chosen = render_distribution_picker(
            key_prefix=f"{key_prefix}_dist",
            initial=dist,
            show_save_section=False,
            require_positive_magnitudes=True,
        )
        return DistributionValue(dist=chosen)

    if vg_type == "VariableRateLoan":
        p = (
            float(getattr(initial, "principal", 300000.0))
            if (initial and getattr(initial, "type", None) == "VariableRateLoan")
            else 300000.0
        )
        r = (
            float(getattr(initial, "initial_rate", 0.065))
            if (initial and getattr(initial, "type", None) == "VariableRateLoan")
            else 0.065
        )
        t = (
            int(getattr(initial, "term_months", 180))
            if (initial and getattr(initial, "type", None) == "VariableRateLoan")
            else 180
        )
        rk = (
            getattr(initial, "rate_key", "mortgage_rate")
            if (initial and getattr(initial, "type", None) == "VariableRateLoan")
            else "mortgage_rate"
        )
        c1, c2, c3 = st.columns(3)
        prin = c1.number_input(
            "Principal ($)",
            value=max(0.0, p),
            min_value=0.0,
            step=5000.0,
            key=f"{key_prefix}_loan_prin",
        )
        rate = c2.number_input(
            "Initial rate (e.g. 0.065)",
            value=r,
            step=0.001,
            format="%.3f",
            key=f"{key_prefix}_loan_rate",
        )
        term = c3.number_input(
            "Term (months)",
            value=t,
            min_value=12,
            max_value=600,
            step=12,
            key=f"{key_prefix}_loan_term",
        )
        rate_key = st.text_input(
            "State key that will hold the current rate", value=rk, key=f"{key_prefix}_loan_rate_key"
        )
        return VariableRateLoanValue(
            principal=float(prin),
            initial_rate=float(rate),
            term_months=int(term),
            rate_key=rate_key,
        )

    if vg_type == "Dividend":
        y = (
            float(getattr(initial, "annual_yield", 0.025))
            if (initial and getattr(initial, "type", None) == "Dividend")
            else 0.025
        )
        k = (
            getattr(initial, "investment_value_key", "portfolio_value")
            if (initial and getattr(initial, "type", None) == "Dividend")
            else "portfolio_value"
        )
        c1, c2 = st.columns(2)
        yld = c1.number_input(
            "Annual yield (e.g. 0.028)",
            value=y,
            step=0.001,
            format="%.3f",
            key=f"{key_prefix}_div_yld",
        )
        key = c2.text_input("Investment value state key", value=k, key=f"{key_prefix}_div_key")
        return DividendValue(annual_yield=float(yld), investment_value_key=key)

    if vg_type == "InvestmentContribution":
        amt = (
            float(getattr(initial, "amount", 500.0))
            if (initial and getattr(initial, "type", None) == "InvestmentContribution")
            else 500.0
        )
        gk = (
            getattr(initial, "growth_key", None)
            if (initial and getattr(initial, "type", None) == "InvestmentContribution")
            else None
        )
        c1, c2 = st.columns(2)
        a = c1.number_input(
            "Contribution amount per period",
            value=max(0.0, amt),
            min_value=0.0,
            step=50.0,
            key=f"{key_prefix}_inv_amt",
        )
        gkey = c2.text_input(
            "Optional growth multiplier key (leave blank for fixed)",
            value=gk or "",
            key=f"{key_prefix}_inv_gk",
        )
        return InvestmentContributionValue(amount=float(a), growth_key=gkey or None)

    if vg_type == "TaxEvent":
        rt = (
            float(getattr(initial, "rate", 0.20))
            if (initial and getattr(initial, "type", None) == "TaxEvent")
            else 0.20
        )
        bk = (
            getattr(initial, "base_key", "taxable_income")
            if (initial and getattr(initial, "type", None) == "TaxEvent")
            else "taxable_income"
        )
        tk = (
            getattr(initial, "tax_key", "tax_paid")
            if (initial and getattr(initial, "type", None) == "TaxEvent")
            else "tax_paid"
        )
        c1, c2, c3 = st.columns(3)
        rate = c1.number_input(
            "Tax rate (e.g. 0.22)", value=rt, step=0.01, format="%.2f", key=f"{key_prefix}_tax_rate"
        )
        base = c2.text_input("Base amount key", value=bk, key=f"{key_prefix}_tax_base")
        taxk = c3.text_input("Output tax accumulator key", value=tk, key=f"{key_prefix}_tax_out")
        return TaxEventValue(rate=float(rate), base_key=base, tax_key=taxk)

    if vg_type == "RateChange":
        # Used primarily by external drivers, but exposed for completeness
        dist = NormalDistribution(mean=0.05, std=0.01)
        if initial and getattr(initial, "type", None) == "RateChange" and hasattr(initial, "dist"):
            dist = initial.dist  # type: ignore[attr-defined]
        chosen = render_distribution_picker(
            key_prefix=f"{key_prefix}_ratechange_dist",
            initial=dist,
            show_save_section=False,
        )
        upk = (
            getattr(initial, "update_key", "some_rate")
            if (initial and getattr(initial, "type", None) == "RateChange")
            else "some_rate"
        )
        up_key = st.text_input(
            "State key to update with the sampled value", value=upk, key=f"{key_prefix}_rate_upkey"
        )
        return RateChangeValue(dist=chosen, update_key=up_key)

    # Fallback
    return FixedValue(value=0.0)


__all__ = [
    "render_value_generator_editor",
    "VG_TYPE_LABELS",
    "get_default_vg",
]
