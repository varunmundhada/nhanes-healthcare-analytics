import streamlit as st
import plotly.express as px
from utils import load_places_state, render_sidebar_controls

st.set_page_config(page_title="Geographic Context (CDC PLACES)", layout="wide")
render_sidebar_controls()
st.title("Geographic Context — model-based estimates, CDC PLACES")

st.error(
    "**This page is NOT NHANES data.** Public-use NHANES carries no geographic "
    "identifier of any kind -- not region, not state, not county; NCHS restricts all "
    "sub-national geographic detail to its Research Data Centers. That is the entire "
    "reason this page exists as a separate data source rather than a simplification "
    "of NHANES down to region-level -- respondents in this dashboard's other pages "
    "were never geolocated at all. Everything below comes from a **separate CDC "
    "product (PLACES)**, a small-area *model-based* estimate derived from BRFSS, not "
    "from measured/interviewed NHANES respondents. Do not read this page as "
    "\"where the NHANES sample lives.\""
)

st.markdown("""
**Provenance of this page's numbers:** CDC PLACES publishes county/place/tract/ZCTA
estimates only -- there is no official PLACES *state*-level file. The state values
shown here were aggregated by this project from the real PLACES county-level dataset
(`data.cdc.gov` resource `i46a-9kgh`, most recent release), using a population-weighted
mean of each state's county crude prevalences. This is a disclosed derivation, not an
official CDC state figure.

Two states (Kentucky, Pennsylvania) came back with no obesity/diabetes values in the
extract this project pulled and are shown as blank/grey on the map -- a known gap in
this specific data pull, not a suppression rule.

PLACES uses BRFSS + Census/ACS inputs from recent years, **not 2017-2018** -- there is
no year-alignment with the NHANES cycle used elsewhere in this dashboard. This page
exists for geographic context only.
""")

places = load_places_state()
measure = st.radio("Measure", ["Obesity (crude %)", "Diabetes (crude %)"], horizontal=True)
col = "obesity_crudeprev_popwt" if measure.startswith("Obesity") else "diabetes_crudeprev_popwt"

fig = px.choropleth(
    places, locations="state_abbr", locationmode="USA-states", color=col,
    scope="usa", color_continuous_scale="OrRd",
    hover_data={"state_name": True, "n_counties": True, col: ":.1f"},
    labels={col: measure},
)
fig.update_layout(title=f"{measure} by state — population-weighted from CDC PLACES county data", margin=dict(t=50))
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Source: CDC PLACES county-level dataset (data.cdc.gov, resource i46a-9kgh), "
    "aggregated to state by this project (population-weighted mean, weight = "
    "PLACES totalpopulation per county). Model-based small-area estimates -- CDC "
    "cautions against using PLACES estimates for program or policy evaluation."
)

with st.expander("Show underlying state table"):
    st.dataframe(places, use_container_width=True)
