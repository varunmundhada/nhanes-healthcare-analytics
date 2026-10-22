import streamlit as st
from utils import (
    load_kpi_long, load_age_standardized, render_sidebar_controls, render_tile,
    small_multiple_bar, INDICATOR_LABELS, FIGURES,
)

st.set_page_config(page_title="Access & Capacity", layout="wide")
render_sidebar_controls()
st.title("Domain 3: Access & Demand-side Capacity")

kpi_long = load_kpi_long()
age_std = load_age_standardized()

L1 = ["UNINSURED", "HAS_USUAL_CARE", "POLYPHARMACY"]

st.subheader("L1 — headline tiles")
cols = st.columns(len(L1))
clicked_indicator = None
for i, ind in enumerate(L1):
    if render_tile(cols[i], kpi_long, ind, key=f"tile_{ind}", age_std=age_std):
        clicked_indicator = ind

if "selected_acc" not in st.session_state:
    st.session_state.selected_acc = "UNINSURED"
if clicked_indicator:
    st.session_state.selected_acc = clicked_indicator

st.divider()
sel = st.session_state.selected_acc
st.subheader(f"L2 — small multiples for: {INDICATOR_LABELS.get(sel, sel)}")
c1, c2 = st.columns(2)
with c1:
    small_multiple_bar(kpi_long, sel, "AGE_BAND", f"{sel} by age band (crude)")
    small_multiple_bar(kpi_long, sel, "RACE_ETH", f"{sel} by race/ethnicity (crude)")
with c2:
    small_multiple_bar(kpi_long, sel, "SEX", f"{sel} by sex (crude)")
    small_multiple_bar(kpi_long, sel, "PIR_Q", f"{sel} by income quintile (crude)")

if sel == "UNINSURED":
    st.info(
        "**UNINSURED has an unusually large RSE (13.7%) and design effect (~18)**: "
        "93.6% of its sampling variance is concentrated in 5 of 15 strata (masked "
        "geography in public-use files). This is a real geographic-concentration "
        "finding, not sampling noise -- see Limitations."
    )

st.divider()
st.subheader("L3 — analytical charts")
st.image(str(FIGURES / "stackedbar_access_by_pir.png"),
          caption="Access & capacity indicators by income quintile. Not mutually exclusive -- additive for visual comparison only. weight=WTINT2YR.")
