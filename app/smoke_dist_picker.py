"""
Quick smoke test for the interactive distribution picker.

Run with:
    streamlit run app/smoke_dist_picker.py
"""

import streamlit as st

from app.components.distribution_viz import render_distribution_picker
from financial_simulator.scenarios import DistributionLibrary

st.set_page_config(page_title="Distribution Picker Smoke", layout="wide")
st.title("🎲 Distribution Picker — Live Demo")

st.markdown("""
This is a standalone smoke test for the core interactive component built in Phase 2.

- Change the distribution type and parameters
- Watch the Plotly preview update live
- Try saving to the in-memory library below
""")

# Simple in-memory library for the smoke
if "library" not in st.session_state:
    st.session_state.library = DistributionLibrary()


def handle_save(saved):
    try:
        st.session_state.library.add(saved)
    except ValueError as e:
        st.warning(str(e))


col1, col2 = st.columns([2, 1])

with col1:
    current_dist = render_distribution_picker(
        key_prefix="smoke",
        library=st.session_state.library,
        on_save_callback=handle_save,
    )

with col2:
    st.markdown("### Current Distribution")
    st.json(current_dist.model_dump(mode="json"), expanded=False)

    st.markdown("### My Library (this session)")
    if st.session_state.library.distributions:
        for d in st.session_state.library.distributions:
            st.write(f"**{d.name}** — `{d.id}`")
            st.caption(d.description or "(no description)")
    else:
        st.info("No distributions saved yet in this session. Use the save section in the picker.")

st.caption("Phase 2 component smoke — part of the financial-simulator scenario builder upgrade.")
