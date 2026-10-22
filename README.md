# NHANES 2017-2018: Design-Based KPI Dashboard and Undiagnosed-Diabetes Screening

Healthcare Analytics mini project (CA3) by **Varun Mundhada**, PRN 23070521184, B.Tech CSE, Symbiosis Institute of Technology, Nagpur.

The project uses the US National Health and Nutrition Examination Survey (NHANES) 2017-2018 public-use files for two things:

1. **Survey-weighted KPI pipeline and dashboard.** It produces 612 estimates for 17 indicators across utilisation, chronic disease and access. Each estimate uses Taylor-linearised standard errors (15 design degrees of freedom), logit confidence intervals and NCHS suppression rules. The pipeline also computes an income concentration index and direct age standardisation. A three-tier Streamlit dashboard reads these estimates.
2. **Screening models for undiagnosed diabetes.** Logistic regression, random forest, gradient boosting and an MLP are trained on 18 non-laboratory features. They use survey weights and nested cross-validation, and are compared with an adapted ADA risk score.

Headline results:
- 18.0% (95% CI 14.8-21.7) of adults with HbA1c >= 6.5% had never been told they had diabetes.
- Lack of insurance is concentrated among poorer adults (concentration index -0.42); undiagnosed diabetes is not (-0.01).
- Logistic regression reached a weighted AUC of 0.788, against 0.734 for the ADA score. The gain is not significant with 150 cases.

## Layout

| Path | What it is |
|---|---|
| `nhanes_kpi_pipeline.py` | Merge, derive indicators, survey-weighted estimation, concentration index -> `out/kpi_long.csv` |
| `age_standardization.py` | Crude vs age-standardised estimates -> `out/age_standardized.csv` |
| `build_evidence_tables.py`, `build_figures.py`, `build_places_choropleth.py` | Evidence tables, dashboard figures, CDC PLACES state aggregation |
| `ml_screening.py` | Nested-CV screening models -> `out/ml_*.csv`, `figures/ml_*.png` |
| `dashboard/` | Streamlit app (`streamlit run dashboard/app.py`) |
| `data/` | NHANES 2017-2018 `.XPT` files (public, CDC/NCHS) and a CDC PLACES extract |
| `IEEE_Paper/` | 6-page IEEE paper (LaTeX source, figures, references, PDF) and the presentation |

## Run

```bash
pip install pandas numpy scipy scikit-learn matplotlib streamlit plotly
python nhanes_kpi_pipeline.py
python age_standardization.py
python build_evidence_tables.py
mkdir -p figures && python build_figures.py
python build_places_choropleth.py
python ml_screening.py            # about 10-15 minutes
streamlit run dashboard/app.py
```

Data source: CDC/NCHS, NHANES 2017-2018, https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx. NHANES public-use files are in the public domain. CDC PLACES is a separate, model-based product and is not NHANES geography.
