# NHANES KPI Dashboard

Three-tier Streamlit dashboard over the NHANES 2017-2018 (_J) analysis in the
parent directory. **This app never recomputes a survey-weighted estimate** — it
reads only the pre-computed CSVs in `../out/`, produced by:

1. `../nhanes_kpi_pipeline.py` → `out/kpi_long.csv`, `out/concentration_index.csv`
2. `../age_standardization.py` → `out/age_standardized.csv`
3. `../build_evidence_tables.py` → `out/t4_concentration_by_band.csv`, `out/t6_suppressed.csv`, `out/fig_*.csv`
4. `../build_places_choropleth.py` → `out/places_state_aggregated.csv`

Run all four (in that order) before launching the dashboard if `out/` is empty
or stale.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Use the sidebar for cycle (only 2017-2018 is
loaded) and crude/age-standardized toggle. Use the left-hand page list for the
three domain pages plus the Geographic Context (CDC PLACES) page.

## Structure

- `app.py` — landing page, dataset-wide summary, suppressed-cell list
- `pages/1_Utilisation_and_Patient_Experience.py`
- `pages/2_Chronic_Disease_Surveillance.py`
- `pages/3_Access_and_Capacity.py`
- `pages/4_Geographic_Context_CDC_PLACES.py` — separate CDC PLACES layer, clearly
  labeled as model-based and NOT geolocated NHANES data
- `utils.py` — shared data loading, tile rendering, suppression-aware charting

## Known limitations surfaced in the UI

- Single NHANES cycle only — no trend/sparkline is fabricated; tiles say so explicitly.
- Suppressed cells (RSE>30% or n<30) render as grey placeholders with a stated reason, never as a silently-blank or zero value.
- Age-standardized toggle does not apply to age-band breakdowns (standardizing removes the age axis by construction) — the UI says this rather than silently ignoring the toggle there.
- The CDC PLACES page aggregates county data to state level itself (no official PLACES state file exists) and says so on the page, along with the two states with missing values in this pull (KY, PA).
