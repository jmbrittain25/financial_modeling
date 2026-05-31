"""
Interactive Distribution Picker with Live Plotly Visualization.

This is the core "wow" component for the Scenario Builder.
Users can:
- Pick any of the 7 distribution types
- Adjust parameters with sliders/number inputs (live)
- See an immediate histogram + analytical PDF (where possible) + statistics
- Save the configured distribution to their personal library

Designed to be dropped into any Streamlit page.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from financial_simulator.core.distributions import (
    AnyDistribution,
    NormalDistribution,
    create_distribution,
)
from financial_simulator.scenarios.models import SavedDistribution

# -----------------------------------------------------------------------------
# Analytical PDF helpers (pure numpy, no scipy dependency)
# -----------------------------------------------------------------------------


def _normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    coeff = 1.0 / (std * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mean) / std) ** 2
    return coeff * np.exp(exponent)


def _uniform_pdf(x: np.ndarray, low: float, high: float) -> np.ndarray:
    width = high - low
    if width <= 0:
        return np.zeros_like(x)
    pdf = np.where((x >= low) & (x <= high), 1.0 / width, 0.0)
    return pdf


def _triangular_pdf(x: np.ndarray, low: float, mode: float, high: float) -> np.ndarray:
    """Piecewise linear triangular PDF (height = 2 / (high-low))."""
    width = high - low
    if width <= 0:
        return np.zeros_like(x)
    height = 2.0 / width
    pdf = np.zeros_like(x, dtype=float)

    # Left side
    left_mask = (x >= low) & (x <= mode)
    if mode > low:
        pdf[left_mask] = height * (x[left_mask] - low) / (mode - low)

    # Right side
    right_mask = (x > mode) & (x <= high)
    if high > mode:
        pdf[right_mask] = height * (high - x[right_mask]) / (high - mode)

    # Exact mode
    pdf[x == mode] = height
    return pdf


def _exponential_pdf(x: np.ndarray, rate: float) -> np.ndarray:
    # rate = lambda
    return np.where(x >= 0, rate * np.exp(-rate * x), 0.0)


def _lognormal_pdf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """PDF of LogNormal(mean=mu of underlying normal, sigma)."""
    if sigma <= 0:
        return np.zeros_like(x)
    # Standard lognormal PDF
    coeff = 1.0 / (x * sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((np.log(x) - mean) / sigma) ** 2
    return np.where(x > 0, coeff * np.exp(exponent), 0.0)


def _beta_pdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Simple Beta PDF using gamma functions (available in numpy)."""
    from math import gamma

    if alpha <= 0 or beta <= 0:
        return np.zeros_like(x)
    B = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
    return np.where(
        (x > 0) & (x < 1),
        (x ** (alpha - 1) * (1 - x) ** (beta - 1)) / B,
        0.0,
    )


def get_analytical_pdf(dist: AnyDistribution, x_grid: np.ndarray) -> np.ndarray | None:
    """Return PDF values on x_grid if analytical form is easy; else None."""
    t = getattr(dist, "type", None)
    try:
        if t == "normal":
            return _normal_pdf(x_grid, dist.mean, dist.std)
        if t == "uniform":
            return _uniform_pdf(x_grid, dist.low, dist.high)
        if t == "triangular":
            return _triangular_pdf(x_grid, dist.low, dist.mode, dist.high)
        if t == "exponential":
            return _exponential_pdf(x_grid, dist.rate)
        if t == "lognormal":
            return _lognormal_pdf(x_grid, dist.mean, dist.sigma)
        if t == "beta":
            return _beta_pdf(x_grid, dist.alpha, dist.beta)
        if t == "constant":
            # Dirac delta - not really plottable as PDF
            return None
    except Exception:
        pass
    return None


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_distribution_preview(
    dist: AnyDistribution,
    title: str | None = None,
    n_samples: int = 6000,
    seed: int = 42,
):
    """Create a rich live preview: histogram + analytical PDF overlay + stats."""
    import plotly.graph_objects as go

    rng = np.random.default_rng(seed)
    samples = np.array([dist.sample(rng) for _ in range(n_samples)])

    fig = go.Figure()

    # Histogram (density normalized)
    fig.add_trace(
        go.Histogram(
            x=samples,
            nbinsx=60,
            name="Samples (density)",
            histnorm="probability density",
            marker_color="#3b82f6",
            opacity=0.65,
        )
    )

    # Analytical PDF (if available)
    x_min, x_max = float(np.min(samples)), float(np.max(samples))
    # Pad a bit for the curve
    pad = (x_max - x_min) * 0.15 if x_max > x_min else 1.0
    x_grid = np.linspace(x_min - pad, x_max + pad, 300)

    pdf = get_analytical_pdf(dist, x_grid)
    if pdf is not None and np.any(pdf > 0):
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=pdf,
                mode="lines",
                name="PDF",
                line=dict(color="#ef4444", width=2.5),
            )
        )

    # Mean and +/- 1 std markers (for applicable dists)
    try:
        mean_val = float(np.mean(samples))
        std_val = float(np.std(samples))
        fig.add_vline(
            x=mean_val,
            line=dict(color="black", width=2, dash="dash"),
            annotation_text="mean",
            annotation_position="top left",
        )
        if std_val > 0:
            fig.add_vline(x=mean_val - std_val, line=dict(color="gray", width=1, dash="dot"))
            fig.add_vline(x=mean_val + std_val, line=dict(color="gray", width=1, dash="dot"))
    except Exception:
        pass

    # Title and layout
    dist_name = dist.__class__.__name__.replace("Distribution", "")
    fig_title = title or f"{dist_name} Distribution (live preview)"
    fig.update_layout(
        title=fig_title,
        xaxis_title="Value",
        yaxis_title="Density",
        height=420,
        bargap=0.05,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    return fig


def get_distribution_stats(
    dist: AnyDistribution, n_samples: int = 8000, seed: int = 42
) -> dict[str, float]:
    """Quick numeric summary for the UI."""
    rng = np.random.default_rng(seed)
    samples = np.array([dist.sample(rng) for _ in range(n_samples)])
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
        "median": float(np.median(samples)),
        "p5": float(np.percentile(samples, 5)),
        "p95": float(np.percentile(samples, 95)),
    }


# -----------------------------------------------------------------------------
# Parameter schemas (for dynamic UI)
# -----------------------------------------------------------------------------

DIST_TYPE_LABELS: dict[str, str] = {
    "normal": "Normal (Gaussian)",
    "uniform": "Uniform",
    "triangular": "Triangular (PERT-style)",
    "lognormal": "Log-Normal",
    "exponential": "Exponential",
    "beta": "Beta (0-1)",
    "constant": "Constant (degenerate)",
}

DIST_TYPE_ORDER = list(DIST_TYPE_LABELS.keys())


def get_default_params(dist_type: str) -> dict[str, Any]:
    """Reasonable starting values for each type."""
    if dist_type == "normal":
        return {"mean": 0.0, "std": 1.0}
    if dist_type == "uniform":
        return {"low": 0.0, "high": 10.0}
    if dist_type == "triangular":
        return {"low": 0.0, "mode": 5.0, "high": 10.0}
    if dist_type == "lognormal":
        return {"mean": 0.0, "sigma": 0.5}
    if dist_type == "exponential":
        return {"rate": 1.0}
    if dist_type == "beta":
        return {"alpha": 2.0, "beta": 5.0}
    if dist_type == "constant":
        return {"value": 1000.0}
    return {}


def render_distribution_picker(
    key_prefix: str = "dist_picker",
    initial: AnyDistribution | None = None,
    library: Any | None = None,  # DistributionLibrary to avoid circular import
    on_save_callback: Callable[[SavedDistribution], None] | None = None,
    show_save_section: bool = True,
    height: int = 420,
):
    """
    Main interactive component.

    Returns the currently configured distribution.
    If the user saves it and on_save_callback is provided, the callback is invoked.
    """
    import streamlit as st

    st.markdown("### 🎲 Distribution Configuration")
    st.caption("Adjust parameters — the preview updates live. Great for exploring uncertainty.")

    # Determine current type
    current_type = getattr(initial, "type", "normal") if initial else "normal"

    # Type selector
    type_label_to_key = {v: k for k, v in DIST_TYPE_LABELS.items()}
    selected_label = st.selectbox(
        "Distribution Type",
        options=list(DIST_TYPE_LABELS.values()),
        index=DIST_TYPE_ORDER.index(current_type) if current_type in DIST_TYPE_ORDER else 0,
        key=f"{key_prefix}_type",
        help="Choose the shape that best represents your uncertainty or variability.",
    )
    dist_type = type_label_to_key[selected_label]

    # Dynamic parameter widgets
    params = get_default_params(dist_type)
    if initial is not None and getattr(initial, "type", None) == dist_type:
        # Seed from existing dist
        for k in params:
            if hasattr(initial, k):
                params[k] = getattr(initial, k)

    cols = st.columns(3 if dist_type in ("triangular", "beta") else 2)

    if dist_type == "normal":
        params["mean"] = cols[0].number_input(
            "Mean (μ)", value=float(params["mean"]), step=0.1, key=f"{key_prefix}_mean"
        )
        params["std"] = cols[1].number_input(
            "Std Dev (σ) > 0",
            value=float(params["std"]),
            min_value=0.001,
            step=0.1,
            key=f"{key_prefix}_std",
        )

    elif dist_type == "uniform":
        params["low"] = cols[0].number_input(
            "Low (minimum)", value=float(params["low"]), step=0.1, key=f"{key_prefix}_low"
        )
        params["high"] = cols[1].number_input(
            "High (maximum) ≥ low", value=float(params["high"]), step=0.1, key=f"{key_prefix}_high"
        )

    elif dist_type == "triangular":
        params["low"] = cols[0].number_input(
            "Low (pessimistic)", value=float(params["low"]), step=0.1, key=f"{key_prefix}_low"
        )
        params["mode"] = cols[1].number_input(
            "Mode (most likely)", value=float(params["mode"]), step=0.1, key=f"{key_prefix}_mode"
        )
        params["high"] = cols[2].number_input(
            "High (optimistic) ≥ mode",
            value=float(params["high"]),
            step=0.1,
            key=f"{key_prefix}_high",
        )

    elif dist_type == "lognormal":
        params["mean"] = cols[0].number_input(
            "Mean of log (μ)", value=float(params["mean"]), step=0.1, key=f"{key_prefix}_mean"
        )
        params["sigma"] = cols[1].number_input(
            "Sigma (σ) > 0",
            value=float(params["sigma"]),
            min_value=0.001,
            step=0.05,
            key=f"{key_prefix}_sigma",
        )

    elif dist_type == "exponential":
        params["rate"] = cols[0].number_input(
            "Rate (λ) > 0",
            value=float(params["rate"]),
            min_value=0.001,
            step=0.1,
            key=f"{key_prefix}_rate",
        )

    elif dist_type == "beta":
        params["alpha"] = cols[0].number_input(
            "Alpha (α) > 0",
            value=float(params["alpha"]),
            min_value=0.01,
            step=0.1,
            key=f"{key_prefix}_alpha",
        )
        params["beta"] = cols[1].number_input(
            "Beta (β) > 0",
            value=float(params["beta"]),
            min_value=0.01,
            step=0.1,
            key=f"{key_prefix}_beta",
        )

    elif dist_type == "constant":
        params["value"] = cols[0].number_input(
            "Constant Value", value=float(params["value"]), step=1.0, key=f"{key_prefix}_value"
        )

    # Build the actual distribution object
    try:
        dist = create_distribution({"type": dist_type, **params})
    except Exception as e:
        st.error(f"Invalid parameters: {e}")
        # Fallback to something safe
        dist = NormalDistribution(mean=0.0, std=1.0)

    # Live preview
    st.markdown("#### Live Preview")
    fig = plot_distribution_preview(dist, n_samples=5000)
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_plot")

    # Stats
    stats = get_distribution_stats(dist)
    cols = st.columns(5)
    cols[0].metric("Mean", f"{stats['mean']:.3f}")
    cols[1].metric("Std Dev", f"{stats['std']:.3f}")
    cols[2].metric("Median", f"{stats['median']:.3f}")
    cols[3].metric("5th %ile", f"{stats['p5']:.3f}")
    cols[4].metric("95th %ile", f"{stats['p95']:.3f}")

    # Optional: Load from library
    if library is not None and hasattr(library, "distributions") and library.distributions:
        st.markdown("#### Load from My Library")
        names = [f"{d.name} ({d.id})" for d in library.distributions]
        choice = st.selectbox(
            "Choose saved distribution", options=["—"] + names, key=f"{key_prefix}_load"
        )
        if choice != "—":
            idx = names.index(choice)
            saved = library.distributions[idx]
            st.info(
                f"Loaded **{saved.name}**. You can tweak the parameters above and save a new copy."
            )
            # Note: we don't auto-switch here to keep the UI simple; user can copy params manually or we could add a "Load into editor" button.

    # Save section
    if show_save_section:
        st.markdown("#### 💾 Save to My Library")
        with st.expander("Save this configuration for reuse in other scenarios", expanded=False):
            save_name = st.text_input(
                "Name", value=f"My {DIST_TYPE_LABELS[dist_type]}", key=f"{key_prefix}_save_name"
            )
            save_desc = st.text_area(
                "Description (optional)", key=f"{key_prefix}_save_desc", height=80
            )
            save_tags = st.text_input("Tags (comma-separated)", key=f"{key_prefix}_save_tags")

            if st.button("Save to Library", type="primary", key=f"{key_prefix}_save_btn"):
                if not save_name.strip():
                    st.warning("Please provide a name.")
                else:
                    saved = SavedDistribution(
                        id=save_name.strip().lower().replace(" ", "-")[:60],
                        name=save_name.strip(),
                        description=save_desc.strip(),
                        dist=dist,
                        tags=[t.strip() for t in save_tags.split(",") if t.strip()],
                    )
                    if on_save_callback:
                        try:
                            on_save_callback(saved)
                            st.success(f"Saved '{saved.name}' to your distribution library!")
                        except Exception as e:
                            st.error(f"Failed to save: {e}")
                    else:
                        st.success(
                            "Distribution ready to be saved (no callback registered in this context)."
                        )
                        st.json(saved.model_dump(mode="json"))

    return dist


# -----------------------------------------------------------------------------
# Simple Preset Gallery (for Step 5 enhancement)
# -----------------------------------------------------------------------------


def render_distribution_gallery(key_prefix: str = "gallery") -> dict[str, AnyDistribution] | None:
    """
    Lightweight gallery of common financial presets.
    Returns the selected distribution if user clicks one, otherwise None.
    """
    import streamlit as st

    from financial_simulator.core.distributions import (
        LogNormalDistribution,
        NormalDistribution,
        TriangularDistribution,
    )

    presets = {
        "Equity Returns (moderate)": NormalDistribution(mean=0.08, std=0.16),
        "Equity Returns (volatile)": NormalDistribution(mean=0.09, std=0.22),
        "Inflation (moderate)": NormalDistribution(mean=0.025, std=0.012),
        "Home Appreciation (triangular)": TriangularDistribution(low=0.01, mode=0.035, high=0.06),
        "Large One-time Expense (lognormal)": LogNormalDistribution(mean=10.5, sigma=0.6),  # ~$36k median
        "Interest Rate Shock (±1.5%)": NormalDistribution(mean=0.0, std=0.015),
    }

    st.markdown("**Common Financial Presets**")
    cols = st.columns(3)
    for i, (label, dist) in enumerate(presets.items()):
        if cols[i % 3].button(label, key=f"{key_prefix}_{i}", use_container_width=True):
            st.toast(f"Loaded preset: {label}")
            return dist
    return None


__all__ = [
    "render_distribution_picker",
    "plot_distribution_preview",
    "get_distribution_stats",
    "DIST_TYPE_LABELS",
    "render_distribution_gallery",
]
