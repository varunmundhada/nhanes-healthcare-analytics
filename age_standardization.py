"""
Task 1d / evidence-pack T2: crude vs. age-standardised estimates.

Reads the analytic file via the pipeline module directly (not from kpi_long.csv)
so both crude and standardised figures come from the same in-memory dataframe in
one pass. Writes out/age_standardized.csv.
"""
import importlib.util
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("pipe", "nhanes_kpi_pipeline.py")
pipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipe)

d = pipe.derive(pipe.build_analytic_file())

INDICATORS = sorted({i for v in pipe.DASHBOARD_SPEC.values() for i in v})
STRATIFIERS = {
    "Overall": None,
    "SEX": "SEX",
    "RACE_ETH": "RACE_ETH",
    "PIR_Q": "PIR_Q",
}

rows = []
for ind in INDICATORS:
    if ind not in d.columns:
        continue
    w = pipe.weight(d, ind)
    if w not in d.columns:
        continue

    for strat_name, col in STRATIFIERS.items():
        if col is None:
            strata_values = [("Overall",)]
        else:
            if col not in d.columns:
                continue
            strata_values = [(v,) for v in sorted(d[col].dropna().unique().astype(str))]

        for (val,) in strata_values:
            if col is None:
                sub = d[d["AGE"] >= 18]
            else:
                sub = d[(d["AGE"] >= 18) & (d[col].astype(str) == val)]

            crude = pipe.taylor_proportion(sub, ind, w)
            std = pipe.age_standardize(sub, ind)

            crude_p = None if crude["suppress"] else round(crude["p"] * 100, 2)
            std_p = None if std["suppressed"] else round(std["std_p"] * 100, 2)

            half_width = None
            flag = False
            if not crude["suppress"] and crude_p is not None and pd.notna(crude["lo"]) and pd.notna(crude["hi"]):
                half_width = round((crude["hi"] - crude["lo"]) * 100 / 2, 2)
                if std_p is not None and half_width is not None:
                    flag = abs(std_p - crude_p) > half_width

            rows.append({
                "indicator": ind,
                "stratifier": strat_name,
                "stratum": val,
                "crude_percent": crude_p,
                "crude_ci_lo": None if crude["suppress"] or pd.isna(crude["lo"]) else round(crude["lo"] * 100, 2),
                "crude_ci_hi": None if crude["suppress"] or pd.isna(crude["hi"]) else round(crude["hi"] * 100, 2),
                "crude_rse_pct": None if pd.isna(crude["rse"]) else round(crude["rse"], 1),
                "crude_n": crude["n"],
                "crude_weighted_pop": None if pd.isna(crude["wpop"]) else round(crude["wpop"]),
                "crude_suppressed": crude["suppress"],
                "std_percent": std_p,
                "std_ci_lo": None if std["suppressed"] or pd.isna(std["std_lo"]) else round(std["std_lo"] * 100, 2),
                "std_ci_hi": None if std["suppressed"] or pd.isna(std["std_hi"]) else round(std["std_hi"] * 100, 2),
                "std_n_age_groups_used": std["n_groups_used"],
                "std_n_age_groups_total": std["n_total"],
                "std_suppressed": std["suppressed"],
                "crude_ci_half_width": half_width,
                "flag_diff_exceeds_half_width": flag,
                "weight_var": w,
            })
    print(f"  [done] {ind}")

out = pd.DataFrame(rows)
out.to_csv("out/age_standardized.csv", index=False)
print(f"\nWrote out/age_standardized.csv ({len(out):,} rows)")
n_flagged = int(out["flag_diff_exceeds_half_width"].sum())
print(f"Flagged (|std-crude| > crude CI half-width): {n_flagged} of {len(out)}")
