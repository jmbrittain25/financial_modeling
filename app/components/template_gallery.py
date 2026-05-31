"""
Template Gallery — Beautiful, card-based selector for high-quality starting scenarios.

Designed to feel professional and help non-technical users quickly find a relevant starting point.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from financial_simulator.scenarios import ScenarioConfig, list_templates, load_template

# Rich metadata for the curated templates (can be expanded later)
TEMPLATE_METADATA: dict[str, dict[str, Any]] = {
    "retirement_30yr": {
        "title": "Retirement Planning (30yr)",
        "category": "Retirement",
        "description": "Long-term portfolio growth with monthly contributions and custom success metrics.",
        "features": ["30-year horizon", "Appreciation process", "Custom metrics"],
        "difficulty": "Beginner",
        "icon": "🏖️",
    },
    "variable_rate_mortgage": {
        "title": "Real Estate + Variable Mortgage",
        "category": "Real Estate",
        "description": "Home purchase with stochastic interest rate driver and VariableRateLoan modeling.",
        "features": ["Interest rate risk", "External driver", "Equity build-up"],
        "difficulty": "Intermediate",
        "icon": "🏠",
    },
    "business_variable_costs": {
        "title": "Small Business Cash Flow",
        "category": "Business Startup",
        "description": "Revenue vs variable operating expenses with triangular uncertainty.",
        "features": ["Variable costs", "Runway analysis", "3-year horizon"],
        "difficulty": "Beginner",
        "icon": "💼",
    },
    "tax_planning_optimized": {
        "title": "Tax-Efficient Portfolio",
        "category": "Tax Planning",
        "description": "Taxable vs tax-advantaged accounts with TaxEventValue and effective tax metrics.",
        "features": ["Tax modeling", "Dual growth buckets", "Tax drag analysis"],
        "difficulty": "Advanced",
        "icon": "🧾",
    },
    "savings_with_growth": {
        "title": "Diversified Portfolio",
        "category": "Portfolio",
        "description": "Multi-asset allocation with GBM, mean-reversion, dividends, and risk metrics.",
        "features": ["Stocks + Bonds + Alts", "Multiple processes", "Drawdown tracking"],
        "difficulty": "Intermediate",
        "icon": "📈",
    },
}


def render_template_gallery(
    key_prefix: str = "gallery",
    on_select: Any | None = None,  # callback(ScenarioConfig)
) -> ScenarioConfig | None:
    """
    Render a nice grid of template cards.
    Returns the selected ScenarioConfig if the user clicks "Load".
    """
    templates = list_templates()

    st.markdown("### 📚 Template Gallery")
    st.caption(
        "High-quality starting points. Load one, then customize freely in the Scenario Builder."
    )

    # Filter / search (simple for now)
    search = st.text_input(
        "Search templates",
        placeholder="retirement, real estate, tax...",
        key=f"{key_prefix}_search",
    )

    filtered = []
    for name in templates:
        meta = TEMPLATE_METADATA.get(name, {})
        title = meta.get("title", name.replace("_", " ").title())
        desc = meta.get("description", "")

        if search and search.lower() not in (title + desc + name).lower():
            continue
        filtered.append(name)

    if not filtered:
        st.warning("No templates match your search.")
        return None

    # Card grid
    cols = st.columns(3)
    selected_cfg = None

    for i, name in enumerate(filtered):
        col = cols[i % 3]
        meta = TEMPLATE_METADATA.get(name, {})

        with col.container(border=True):
            icon = meta.get("icon", "📄")
            title = meta.get("title", name.replace("_", " ").title())
            desc = meta.get("description", "A solid starting scenario.")
            features = meta.get("features", [])
            difficulty = meta.get("difficulty", "Intermediate")

            st.markdown(f"### {icon} {title}")
            st.caption(f"**{difficulty}** • {meta.get('category', 'General')}")

            st.write(desc)

            if features:
                st.markdown("**Key Features:**")
                for f in features:
                    st.write(f"• {f}")

            if st.button(
                "Load into Builder",
                key=f"{key_prefix}_load_{name}",
                type="primary",
                use_container_width=True,
            ):
                try:
                    cfg = load_template(name)
                    st.success(f"Loaded **{title}**")
                    if on_select:
                        on_select(cfg)
                    selected_cfg = cfg
                except Exception as e:
                    st.error(f"Failed to load template: {e}")

    return selected_cfg


__all__ = ["render_template_gallery", "TEMPLATE_METADATA"]
