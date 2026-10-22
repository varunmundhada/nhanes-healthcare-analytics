"""
Shared data loading and rendering helpers for the NHANES KPI dashboard.

HARD RULE: this module and every page under pages/ read pre-computed CSVs
from ../out/ only. Nothing here calls taylor_proportion(), age_standardize(),
or any other estimator live -- every number on screen was computed offline by
nhanes_kpi_pipeline.py / age_standardization.py / build_evidence_tables.py.
"""
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
OUT = ROOT / "out"
FIGURES = ROOT / "figures"

CYCLE_LABEL = "NHANES 2017-2018 (_J)"  # only cycle this dashboard has data for


@st.cache_data
def load_kpi_long():
    return pd.read_csv(OUT / "kpi_long.csv")


@st.cache_data
def load_age_standardized():
    return pd.read_csv(OUT / "age_standardized.csv")


@st.cache_data
def load_concentration_pooled():
    return pd.read_csv(OUT / "concentration_index.csv")


@st.cache_data
def load_concentration_by_band():
    return pd.read_csv(OUT / "t4_concentration_by_band.csv")


@st.cache_data
def load_suppressed():
    return pd.read_csv(OUT / "t6_suppressed.csv")


@st.cache_data
def load_places_state():
    return pd.read_csv(OUT / "places_state_aggregated.csv")


INDICATOR_LABELS = {
    "HEALTH_FAIRPOOR": "Fair/poor self-rated health",
    "ADMITTED_12M": "Hospital admission, past 12mo",
    "HIGH_UTIL": "High healthcare utilisation (4+ contacts/yr)",
    "HAS_USUAL_CARE": "Has a usual source of care",
    "USUAL_CARE_IS_ED": "Usual source of care is the ED",
    "DM_ANY": "Diabetes (any, diagnosed or undiagnosed)",
    "HTN_ANY": "Hypertension (any, ACC/AHA 130/80)",
    "OBESE": "Obesity (BMI>=30)",
    "DM_UNDIAG": "Undiagnosed diabetes (among diabetics)",
    "HTN_UNDIAG": "Undiagnosed hypertension, 130/80 (among hypertensives)",
    "HTN_UNDIAG_130": "Undiagnosed hypertension, 130/80",
    "HTN_UNDIAG_140": "Undiagnosed hypertension, 140/90",
    "HTN_UNDIAG_BAND": "Undiagnosed, reclassified band only (130-139/80-89)",
    "PREDM_LAB": "Prediabetes (HbA1c 5.7-6.4%)",
    "DEPRESSION": "Depression (PHQ-9 >=10)",
    "UNINSURED": "Uninsured",
    "POLYPHARMACY": "Polypharmacy (5+ medications)",
}


def suppression_reason(row) -> str:
    reasons = []
    if row.get("unweighted_n", 999) < 30:
        reasons.append("n<30")
    rse = row.get("rse_pct")
    if pd.notna(rse) and rse > 30:
        reasons.append("RSE>30%")
    return " & ".join(reasons) if reasons else "suppressed"


def render_tile(col, kpi_long: pd.DataFrame, indicator: str, key: str, age_std: pd.DataFrame = None):
    """One L1 tile. Returns True if this tile was just clicked.
    If st.session_state.std_toggle asks for age-standardized and age_std is
    supplied, the headline number switches to the standardized estimate --
    still sourced from a pre-computed CSV, never recomputed here."""
    label = INDICATOR_LABELS.get(indicator, indicator)
    use_std = st.session_state.get("std_toggle", "Crude").startswith("Age")

    row = kpi_long[(kpi_long.indicator == indicator) & (kpi_long.stratifier == "Overall")]
    with col:
        if row.empty:
            st.warning(f"{label}: not available (source file not downloaded)")
            return False
        r = row.iloc[0]

        std_row = None
        if use_std and age_std is not None:
            m = age_std[(age_std.indicator == indicator) & (age_std.stratifier == "Overall")]
            if not m.empty:
                std_row = m.iloc[0]

        if use_std and std_row is not None and not std_row["std_suppressed"]:
            pct = std_row["std_percent"]
            ci_txt = f"{std_row['std_ci_lo']:.1f}-{std_row['std_ci_hi']:.1f}%"
            extra = f"Age groups used: {std_row['std_n_age_groups_used']}/{std_row['std_n_age_groups_total']} (z=1.96 CI, not t(15))"
        elif r["suppressed"]:
            st.markdown(
                f"<div style='background:#e0e0e0;border-radius:8px;padding:14px;text-align:center;'>"
                f"<b>{label}</b><br><span style='color:#666;'>Suppressed<br>"
                f"({suppression_reason(r)})</span></div>",
                unsafe_allow_html=True,
            )
            st.button("Details unavailable", key=key, disabled=True)
            return False
        else:
            pct = r["percent"]
            ci_txt = f"{r['ci_lo_per_100k']/1000:.1f}-{r['ci_hi_per_100k']/1000:.1f}%"
            extra = f"dof={r.get('dof', 'NA')}"

        pop_m = r["weighted_pop"] / 1_000_000 if pd.notna(r["weighted_pop"]) else None
        pop_str = f"~{pop_m:.1f}M adults" if pop_m else ""
        help_text = (
            f"{pct}% (95% CI {ci_txt})\n"
            f"RSE: {r['rse_pct']}%  |  n={r['unweighted_n']}  |  {extra}\n"
            f"Weight: {r['weight_var']}  |  {pop_str}\n"
            f"Trend: not available -- single cycle only (NHANES 2017-2018)"
        )
        clicked = st.button(
            f"**{label}**\n\n# {pct}%\n\n{'age-standardized' if use_std and std_row is not None else 'crude'}",
            key=key, help=help_text, use_container_width=True,
        )
        st.caption(f"CI {ci_txt} | RSE {r['rse_pct']}% | n={r['unweighted_n']}")
        return clicked


def render_sidebar_controls():
    """Global controls shared across every page. Persisted via session_state."""
    st.sidebar.header("Global controls")
    st.sidebar.selectbox(
        "Cycle", options=[CYCLE_LABEL],
        help="Only one cycle was loaded for this build. A cycle selector is provided "
             "for future extension, but there is nothing to switch to yet.",
        key="cycle",
    )
    st.sidebar.radio(
        "Estimate type", options=["Crude", "Age-standardized (2000 US std. pop.)"],
        key="std_toggle",
        help="Age-standardized uses direct standardization to 8 adult age groups "
             "(see Methodology). CI method differs: crude uses t(dof=15); "
             "standardized uses a normal (z=1.96) approximation.",
    )
    st.sidebar.caption(
        "Data: NHANES 2017-2018 (_J), survey-weighted, Taylor-linearised SE. "
        "Suppressed per NCHS standards (RSE>30% or n<30)."
    )


def small_multiple_bar(kpi_long: pd.DataFrame, indicator: str, stratifier: str, title: str):
    import plotly.graph_objects as go
    sub = kpi_long[(kpi_long.indicator == indicator) & (kpi_long.stratifier == stratifier)].copy()
    if sub.empty:
        st.info(f"No data for {indicator} by {stratifier}")
        return
    sub["display_pct"] = sub["percent"]
    colors = ["#cbd5e0" if s else "#2b6cb0" for s in sub["suppressed"]]
    text = [("suppressed" if s else f"{p:.1f}%") for s, p in zip(sub["suppressed"], sub["percent"])]
    hover = [
        (f"Suppressed: {suppression_reason(row)}" if row["suppressed"] else
         f"{row['percent']}% (CI {row['ci_lo_per_100k']/1000:.1f}-{row['ci_hi_per_100k']/1000:.1f}%)<br>"
         f"RSE {row['rse_pct']}% | n={row['unweighted_n']}")
        for _, row in sub.iterrows()
    ]
    fig = go.Figure(go.Bar(x=sub["stratum"], y=sub["percent"].fillna(0), marker_color=colors,
                            text=text, textposition="outside", hovertext=hover, hoverinfo="text"))
    fig.update_layout(title=title, yaxis_title="Percent", margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
