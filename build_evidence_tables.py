"""
Task 2 support: builds every CSV feeding results_tables.xlsx and figures/.
Run after nhanes_kpi_pipeline.py and age_standardization.py.
"""
import importlib.util
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("pipe", "nhanes_kpi_pipeline.py")
pipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipe)

d = pipe.derive(pipe.build_analytic_file())
kpi_long = pd.read_csv("out/kpi_long.csv")

# ---------------------------------------------------------------------------
# T4b: within-age-band concentration index (pooled figures already in
# out/concentration_index.csv -- this adds the by-band breakdown)
# ---------------------------------------------------------------------------
CI_INDICATORS = ["UNINSURED", "DM_UNDIAG", "DM_ANY", "HTN_ANY", "HTN_UNDIAG",
                  "HTN_UNDIAG_130", "HTN_UNDIAG_140", "HTN_UNDIAG_BAND",
                  "OBESE", "HEALTH_FAIRPOOR", "DEPRESSION", "POLYPHARMACY"]
AGE_BANDS_FOR_CI = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]

rows = []
pooled = pd.read_csv("out/concentration_index.csv").set_index("indicator")["concentration_index_income"].to_dict()
for ind in CI_INDICATORS:
    if ind not in d.columns:
        continue
    rows.append({"indicator": ind, "band": "Pooled (all adults)",
                 "concentration_index": pooled.get(ind, np.nan)})
    for band in AGE_BANDS_FOR_CI:
        sub = d[d.AGE_BAND == band]
        ci = pipe.concentration_index(sub, ind, min_age=0)  # sub already age-restricted by band
        n = len(sub[[ind]].dropna())
        rows.append({"indicator": ind, "band": band,
                     "concentration_index": ci, "n_in_band": n})

ci_by_band = pd.DataFrame(rows)
ci_by_band.to_csv("out/t4_concentration_by_band.csv", index=False)
print(f"[T4] out/t4_concentration_by_band.csv ({len(ci_by_band)} rows)")

# ---------------------------------------------------------------------------
# T5: diagnosis gap, all threshold/definition variants in one table
# ---------------------------------------------------------------------------
lab_pos = d[(d["AGE"] >= 18) & (d["DM_LAB"] == 1)].copy()
lab_pos["UNDIAG_A"] = lab_pos["DM_UNDIAG"]
lab_pos["UNDIAG_B"] = np.where(lab_pos["DIQ010"] == 3, 1.0, lab_pos["DM_UNDIAG"])

t5_rows = []
for label, col, note in [
    ("DM_UNDIAG (borderline=diagnosed, as-shipped)", "UNDIAG_A",
     "DIQ010 code 3 (borderline) treated as diagnosed"),
    ("DM_UNDIAG (borderline=undiagnosed)", "UNDIAG_B",
     "DIQ010 code 3 (borderline) treated as undiagnosed"),
]:
    r = pipe.taylor_proportion(lab_pos, col, "WTMEC2YR")
    t5_rows.append({"measure": label, "percent": round(r["p"]*100, 2),
                     "ci_lo": round(r["lo"]*100, 2), "ci_hi": round(r["hi"]*100, 2),
                     "rse_pct": round(r["rse"], 1), "n": r["n"], "dof": r["dof"],
                     "weighted_pop": round(r["wpop"]), "note": note})

for ind, note in [
    ("HTN_UNDIAG_130", "2017 ACC/AHA threshold (130/80)"),
    ("HTN_UNDIAG_140", "pre-2017 threshold (140/90)"),
    ("HTN_UNDIAG_BAND", "reclassified band only: 130-139/80-89, excluded from 140/90 group"),
]:
    row = kpi_long[(kpi_long.stratifier == "Overall") & (kpi_long.indicator == ind)].iloc[0]
    t5_rows.append({"measure": ind, "percent": row["percent"],
                     "ci_lo": row["ci_lo_per_100k"]/1000, "ci_hi": row["ci_hi_per_100k"]/1000,
                     "rse_pct": row["rse_pct"], "n": row["unweighted_n"], "dof": row["dof"],
                     "weighted_pop": row["weighted_pop"], "note": note})

t5 = pd.DataFrame(t5_rows)
t5.to_csv("out/t5_diagnosis_gap.csv", index=False)
print(f"[T5] out/t5_diagnosis_gap.csv ({len(t5)} rows)")

# ---------------------------------------------------------------------------
# T6: suppressed cells with reason
# ---------------------------------------------------------------------------
supp = kpi_long[kpi_long.suppressed == True].copy()
def reason(row):
    r = []
    if row["unweighted_n"] < 30:
        r.append("n<30")
    if pd.notna(row["rse_pct"]) and row["rse_pct"] > 30:
        r.append("RSE>30%")
    return " & ".join(r) if r else "unknown"
supp["reason"] = supp.apply(reason, axis=1)
t6 = supp[["indicator", "stratifier", "stratum", "unweighted_n", "rse_pct", "reason"]]
t6.to_csv("out/t6_suppressed.csv", index=False)
print(f"[T6] out/t6_suppressed.csv ({len(t6)} rows)")

# ---------------------------------------------------------------------------
# T7: validation vs published (compiled from Task 1f fetches)
# ---------------------------------------------------------------------------
t7 = pd.DataFrame([
    {"indicator": "OBESE", "mine_crude": 41.95, "mine_standardized": 41.80,
     "published": "42.5 (crude) / 42.4 (age-adj)", "published_age_range": "20+",
     "source": "Hales et al., NCHS Data Brief 360 (2020)",
     "url": "https://www.cdc.gov/nchs/products/databriefs/db360.htm",
     "definitional_differences": "Theirs: age 20+, 3-group (20-39/40-59/60+) standardization to 2000 Census pop. Mine: age 18+, 8-group standardization to 2000 standard million. Both use measured BMI>=30, same NHANES 2017-2018 cycle."},
    {"indicator": "HTN_ANY", "mine_crude": 48.40, "mine_standardized": 45.00,
     "published": "45.4 (age-adjusted)", "published_age_range": "18+",
     "source": "Ostchega et al., NCHS Data Brief 364 (2020)",
     "url": "https://www.cdc.gov/nchs/products/databriefs/db364.htm",
     "definitional_differences": "Theirs includes 'currently taking BP medication' as an independent qualifying criterion; mine relies on measured BP (130/80) OR self-report (BPQ020) only, no direct medication-use variable. Same cycle, same 130/80 threshold."},
    {"indicator": "DM_ANY", "mine_crude": 13.40, "mine_standardized": 13.51,
     "published": "13.2 (age-adjusted, 95% CI 12.0-14.5)", "published_age_range": "18+",
     "source": "National Diabetes Statistics Report, Appendix Table 1",
     "url": "https://archive.cdc.gov/www_cdc_gov/diabetes/php/data-research/appendix.html",
     "definitional_differences": "Theirs pools 2017-March 2020 (more cycles, larger n). Mine is 2017-2018 only AND omits fasting plasma glucose entirely (GLU_J not downloaded) -- case-finding is HbA1c+self-report only. Counter-intuitively mine is not below theirs despite the narrower case definition; flagged as an open question, not resolved."},
    {"indicator": "UNINSURED", "mine_crude": 13.89, "mine_standardized": 15.58,
     "published": "~12.5 (adults 18-64, Jan-Jun 2018, at time of interview) -- TODO-CITE, not independently verified",
     "published_age_range": "18-64",
     "source": "CDC/NCHS NHIS Early Release Program (provisional)",
     "url": "https://archive.cdc.gov/www_cdc_gov/nchs/data/nhis/earlyrelease/insur201811.pdf",
     "definitional_differences": "Different survey entirely (NHIS vs NHANES) -- different sampling frame, questionnaire wording, and mode. Age range mismatch: 18-64 vs my 18+ (which includes near-universally-insured 65+ and should pull mine down, yet mine is higher). PDF could not be opened directly in this session (CDC WAF blocked every fetch attempt) -- number is from search-engine synthesis only, not a primary-source read. Treat as provisional pending direct verification."},
])
t7.to_csv("out/t7_validation.csv", index=False)
print(f"[T7] out/t7_validation.csv ({len(t7)} rows)")

# ---------------------------------------------------------------------------
# Figure data: heatmap (AGE_BAND x CC_BAND -> ADMITTED_12M rate)
# ---------------------------------------------------------------------------
heat_rows = []
for age_band in ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]:
    for cc_band in ["0", "1", "2", "3+"]:
        sub = d[(d.AGE_BAND == age_band) & (d.CC_BAND == cc_band) & (d.AGE >= 18)]
        r = pipe.taylor_proportion(sub, "ADMITTED_12M", "WTINT2YR")
        heat_rows.append({"age_band": age_band, "cc_band": cc_band,
                           "rate_per_100k": None if r["suppress"] else round(r["p"]*100000, 1),
                           "n": r["n"], "suppressed": r["suppress"]})
heat = pd.DataFrame(heat_rows)
heat.to_csv("out/fig_heatmap_admission_by_age_cc.csv", index=False)
print(f"[fig] out/fig_heatmap_admission_by_age_cc.csv ({len(heat)} rows)")

# ---------------------------------------------------------------------------
# Figure data: scatter (HIGH_UTIL vs HEALTH_FAIRPOOR, cell = AGE_BAND x SEX, bubble = weighted_pop)
# ---------------------------------------------------------------------------
scatter_rows = []
for age_band in ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]:
    for sex in ["Male", "Female"]:
        sub = d[(d.AGE_BAND == age_band) & (d.SEX == sex) & (d.AGE >= 18)]
        r_util = pipe.taylor_proportion(sub, "HIGH_UTIL", "WTINT2YR")
        r_health = pipe.taylor_proportion(sub, "HEALTH_FAIRPOOR", "WTINT2YR")
        scatter_rows.append({
            "age_band": age_band, "sex": sex,
            "high_util_pct": None if r_util["suppress"] else round(r_util["p"]*100, 2),
            "health_fairpoor_pct": None if r_health["suppress"] else round(r_health["p"]*100, 2),
            "weighted_pop": r_util["wpop"], "n": r_util["n"],
        })
scatter = pd.DataFrame(scatter_rows)
scatter.to_csv("out/fig_scatter_util_vs_health.csv", index=False)
print(f"[fig] out/fig_scatter_util_vs_health.csv ({len(scatter)} rows)")

# ---------------------------------------------------------------------------
# Figure data: population pyramid (rate-based) -- HTN_ANY by AGE_BAND x SEX
# ---------------------------------------------------------------------------
pyr_rows = []
for age_band in ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]:
    for sex in ["Male", "Female"]:
        sub = d[(d.AGE_BAND == age_band) & (d.SEX == sex) & (d.AGE >= 18)]
        r = pipe.taylor_proportion(sub, "HTN_ANY", "WTMEC2YR")
        pyr_rows.append({"age_band": age_band, "sex": sex,
                          "rate_pct": None if r["suppress"] else round(r["p"]*100, 2), "n": r["n"]})
pyr = pd.DataFrame(pyr_rows)
pyr.to_csv("out/fig_pyramid_htn_by_age_sex.csv", index=False)
print(f"[fig] out/fig_pyramid_htn_by_age_sex.csv ({len(pyr)} rows)")

# ---------------------------------------------------------------------------
# Figure data: stacked bar -- Domain 3 access/capacity indicators by PIR_Q
# ---------------------------------------------------------------------------
access_rows = []
for ind in ["UNINSURED", "HAS_USUAL_CARE", "POLYPHARMACY"]:
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        sub = d[(d.PIR_Q == q) & (d.AGE >= 18)]
        w = pipe.weight(d, ind)
        r = pipe.taylor_proportion(sub, ind, w)
        access_rows.append({"indicator": ind, "pir_q": q,
                             "percent": None if r["suppress"] else round(r["p"]*100, 2), "n": r["n"]})
access = pd.DataFrame(access_rows)
access.to_csv("out/fig_stackedbar_access_by_pir.csv", index=False)
print(f"[fig] out/fig_stackedbar_access_by_pir.csv ({len(access)} rows)")

# ---------------------------------------------------------------------------
# Figure data: treemap -- weighted population "burden share" by condition
# ---------------------------------------------------------------------------
burden_rows = []
for ind, label in [("DM_ANY", "Diabetes"), ("HTN_ANY", "Hypertension"), ("OBESE", "Obesity"),
                    ("HF", "Heart failure"), ("CHD", "Coronary heart disease"),
                    ("COPD", "COPD"), ("DEPRESSION", "Depression")]:
    if ind not in d.columns:
        continue
    w = pipe.weight(d, ind)
    r = pipe.taylor_proportion(d[d.AGE >= 18], ind, w)
    burden_rows.append({"condition": label, "indicator": ind,
                         "weighted_pop_with_condition": None if r["suppress"] else round(r["wpop"]*r["p"]),
                         "percent": None if r["suppress"] else round(r["p"]*100, 2), "n": r["n"]})
burden = pd.DataFrame(burden_rows)
burden.to_csv("out/fig_treemap_burden.csv", index=False)
print(f"[fig] out/fig_treemap_burden.csv ({len(burden)} rows)")

print("\nAll Task 2 supporting tables written to out/.")
