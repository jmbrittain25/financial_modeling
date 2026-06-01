"""
Quick smoke test for the interactive external driver editor.

Run with:
    streamlit run app/smoke_driver_editor.py
"""

import sys
from pathlib import Path

# Ensure project root is on path for `from app.components...` imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.components.driver_viz import render_external_driver_editor
from financial_simulator.scenarios.drivers import make_interest_rate_driver

st.set_page_config(page_title="External Driver Editor Smoke", layout="wide")
st.title("🔗 External Driver Editor — Live Demo")
st.markdown("""
This is a standalone smoke test for the new driver component (Phase 2).

- Change driver type and parameters — the path preview updates live.
- Discrete drivers embed the full distribution picker.
- The returned object is a valid `AnyExternalDriver` ready for ScenarioConfig.
""")

if "demo_driver" not in st.session_state:
    st.session_state.demo_driver = make_interest_rate_driver()


def handle_use(driver):
    st.session_state.demo_driver = driver
    st.toast(f"Driver captured: {driver.name}")


driver = render_external_driver_editor(
    key_prefix="smoke_driver",
    initial=st.session_state.demo_driver,
    on_save_callback=handle_use,
)

st.divider()
st.subheader("Resulting Driver Object (ready for ScenarioConfig.external_drivers)")
st.json(driver.model_dump(mode="json"), expanded=False)

st.caption("You can copy the JSON and paste it into a ScenarioConfig or template.")
