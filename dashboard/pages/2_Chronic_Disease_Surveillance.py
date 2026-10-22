import streamlit as st
from utils import (
    load_kpi_long, load_age_standardized, load_concentration_pooled, load_concentration_by_band,
    render_sidebar_controls, render_tile, small_multiple_bar, INDICATOR_LABELS, FIGURES,
)

st.set_page_config(page_title="Chronic Disease Surveillance", layout="wide")
render_sidebar_controls()
st.title("Domain 2: Chronic Disease Surveillance")

kpi_long = load_kpi_long()
age_std = load_age_standardized()
ci_pooled = load_concentration_pooled()
ci_band = load_concentration_by_band()

L1 = ["DM_ANY", "HTN_ANY", "OBESE"]
L2_EXTRA = ["DM_UNDIAG", "HTN_UNDIAG_130", "HTN_UNDIAG_140", "HTN_UNDIAG_BAND", "PREDM_LAB", "DEPRESSION"]

st.subheader("L1 — headline tiles")
cols = st.columns(len(L1))
clicked_indicator = None
for i, ind in enumerate(L1):
    if render_tile(cols[i], kpi_long, ind, key=f"tile_{ind}", age_std=age_std):
        clicked_indicator = ind

st.subheader("L2 tiles — the diagnosis gap")
st.warning(
    "**HTN_UNDIAG at either threshold is not a clean detection-failure measure**: "
    "NHANES measures BP once, while clinical diagnosis requires elevation on 2+ visits, "
    "so transient elevation inflates both HTN_UNDIAG_130 and HTN_UNDIAG_140. "
    "DM_UNDIAG (HbA1c-based, no white-coat effect) is the more solid single estimate."
)
cols2 = st.columns(len(L2_EXTRA))
for i, ind in enumerate(L2_EXTRA):
    if render_tile(cols2[i], kpi_long, ind, key=f"tile2_{ind}", age_std=age_std):
        clicked_indicator = ind

if "selected_dis" not in st.session_state:
    st.session_state.selected_dis = "DM_ANY"
if clicked_indicator:
    st.session_state.selected_dis = clicked_indicator

st.divider()
sel = st.session_state.selected_dis
st.subheader(f"Breakdown for: {INDICATOR_LABELS.get(sel, sel)}")
c1, c2 = st.columns(2)
with c1:
    small_multiple_bar(kpi_long, sel, "AGE_BAND", f"{sel} by age band (crude)")
    small_multiple_bar(kpi_long, sel, "RACE_ETH", f"{sel} by race/ethnicity (crude)")
with c2:
    small_multiple_bar(kpi_long, sel, "SEX", f"{sel} by sex (crude)")
    small_multiple_bar(kpi_long, sel, "PIR_Q", f"{sel} by income quintile (crude)")

st.divider()
st.subheader("Equity gradient: concentration index (income)")
st.caption("CI < 0 = concentrated among the poor. Pooled figure is confounded by age composition for age-graded disease -- see within-band panel.")
c3, c4 = st.columns(2)
with c3:
    st.markdown("**Pooled (all adults 18+)**")
    row = ci_pooled[ci_pooled.indicator == sel]
    if not row.empty:
        st.metric(sel, f"{row.iloc[0]['concentration_index_income']:.4f}")
with c4:
    st.markdown("**Within age band** (tests whether the pooled figure is an age-composition artifact)")
    band_rows = ci_band[(ci_band.indicator == sel) & (ci_band.band != "Pooled (all adults)")]
    if not band_rows.empty:
        st.dataframe(band_rows[["band", "concentration_index", "n_in_band"]], hide_index=True, use_container_width=True)
    else:
        st.info("Not computed for this indicator.")

st.divider()
st.subheader("L3 — analytical charts")
c5, c6 = st.columns(2)
with c5:
    st.image(str(FIGURES / "heatmap_admission_age_cc.png"),
              caption="Admission rate by age x multimorbidity. Source: NHANES 2017-2018, weight=WTINT2YR.")
with c6:
    st.image(str(FIGURES / "pyramid_htn_rate_by_age_sex.png"),
              caption="Rate-based pyramid: hypertension prevalence by age/sex. weight=WTMEC2YR.")
st.image(str(FIGURES / "treemap_burden_share.png"),
          caption="Weighted population burden share by condition. NO cost/budget data in NHANES.")
st.image(str(FIGURES / "barinbar_vs_hp2030.png"),
          caption="This study vs. Healthy People 2030 baseline/target (HTN_ANY, OBESE, UNINSURED). Diabetes omitted -- no HP2030 objective measures prevalence.")
