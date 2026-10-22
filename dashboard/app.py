import streamlit as st
from utils import (
    load_kpi_long, load_age_standardized, load_suppressed, render_sidebar_controls,
    INDICATOR_LABELS, CYCLE_LABEL,
)

st.set_page_config(page_title="NHANES KPI Dashboard", layout="wide", page_icon="📊")
render_sidebar_controls()

st.title("NHANES Three-Tier KPI Dashboard")
st.caption(f"{CYCLE_LABEL} — survey-weighted, Taylor-linearised SE, design dof=15")

st.markdown("""
This dashboard reads **only** pre-computed CSVs under `out/` — every percentage,
confidence interval, RSE, and n was calculated offline by `nhanes_kpi_pipeline.py`,
`age_standardization.py`, and `build_evidence_tables.py`. **No estimate is
recomputed inside this app.**

Use the sidebar to switch between crude and age-standardized estimates. Use the
page list on the left to open each of the three domain pages (Utilisation &
Patient Experience / Chronic Disease Surveillance / Access & Capacity), or the
Geographic Context page for the CDC PLACES choropleth layer.
""")

kpi_long = load_kpi_long()
age_std = load_age_standardized()
supp = load_suppressed()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Indicators tracked", kpi_long["indicator"].nunique())
col2.metric("Total estimate rows", len(kpi_long))
col3.metric("Suppressed cells", int(kpi_long["suppressed"].sum()),
            help="RSE>30% or unweighted n<30, per NCHS Data Presentation Standards")
col4.metric("Design degrees of freedom", "15", help="~15 strata x 2 PSUs; t(0.975,15)=2.131 used for crude CIs")

st.divider()
st.subheader("What's in each domain page")
st.markdown("""
- **Utilisation & Patient Experience**: self-rated health, admissions, high utilisation, usual source of care
- **Chronic Disease Surveillance**: diabetes, hypertension (both thresholds), obesity, prediabetes, depression, undiagnosed-disease gaps
- **Access & Capacity**: insurance coverage, polypharmacy, usual source of care
- **Geographic Context (CDC PLACES)**: a clearly separate, model-based state layer — NHANES respondents are **not** geolocated in public-use files
""")

st.subheader("Suppressed cells (all domains)")
st.caption("Rendered as grey placeholders on every chart below; listed here for completeness.")
st.dataframe(supp, use_container_width=True, height=250)

st.divider()
st.caption(
    "Limitations that apply to every page: cross-sectional design (no incidence/causality); "
    "single-visit BP measurement vs. 2-visit clinical diagnosis; HbA1c-only diabetes "
    "ascertainment (GLU_J not available); RXQ_RX placeholder-row bug found and fixed "
    "(see Methodology); no geography in public-use files."
)
