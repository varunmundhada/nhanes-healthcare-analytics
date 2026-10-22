import streamlit as st
from utils import (
    load_kpi_long, load_age_standardized, render_sidebar_controls, render_tile,
    small_multiple_bar, INDICATOR_LABELS, FIGURES,
)

st.set_page_config(page_title="Utilisation & Patient Experience", layout="wide")
render_sidebar_controls()
st.title("Domain 1: Utilisation & Patient Experience")

kpi_long = load_kpi_long()
age_std = load_age_standardized()

L1 = ["HEALTH_FAIRPOOR", "ADMITTED_12M", "HIGH_UTIL"]
L2_EXTRA = ["HAS_USUAL_CARE", "USUAL_CARE_IS_ED"]

st.subheader("L1 — headline tiles (click one to filter the breakdown below)")
cols = st.columns(len(L1) + len(L2_EXTRA))
clicked_indicator = None
for i, ind in enumerate(L1 + L2_EXTRA):
    if render_tile(cols[i], kpi_long, ind, key=f"tile_{ind}", age_std=age_std):
        clicked_indicator = ind

if "selected_util" not in st.session_state:
    st.session_state.selected_util = "HEALTH_FAIRPOOR"
if clicked_indicator:
    st.session_state.selected_util = clicked_indicator

st.divider()
sel = st.session_state.selected_util
st.subheader(f"L2 — small multiples for: {INDICATOR_LABELS.get(sel, sel)}")
st.caption("Age-standardized toggle does not apply to the age-band panel (standardizing removes the age axis by construction).")
c1, c2 = st.columns(2)
with c1:
    small_multiple_bar(kpi_long, sel, "AGE_BAND", f"{sel} by age band (crude)")
    small_multiple_bar(kpi_long, sel, "RACE_ETH", f"{sel} by race/ethnicity (crude)")
with c2:
    small_multiple_bar(kpi_long, sel, "SEX", f"{sel} by sex (crude)")
    small_multiple_bar(kpi_long, sel, "PIR_Q", f"{sel} by income quintile (crude)")

st.divider()
st.subheader("L3 — analytical charts")
st.image(str(FIGURES / "scatter_util_vs_health.png"),
          caption="Utilisation vs. self-rated health by age/sex, bubble=weighted population. Source: NHANES 2017-2018, weight=WTINT2YR.")
