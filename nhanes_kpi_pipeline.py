"""
NHANES three-tier KPI pipeline
==============================
Produces a tidy CSV that a dashboard (Power BI / Tableau / Streamlit) reads directly.

WHY THIS FILE EXISTS
--------------------
NHANES is a complex multistage probability sample. Counting rows gives you the
WRONG answer. Every estimate must use the sample weights, and every standard
error must account for the stratified cluster design (SDMVSTRA / SDMVPSU).
This script does both, using Taylor-series linearisation for a ratio estimator.

USAGE
-----
1. Download the .XPT files you need from
   https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx
   and drop them in ./data/  (keep the CDC filenames, e.g. DEMO_J.XPT)
2. python nhanes_kpi_pipeline.py
3. Output: ./out/kpi_long.csv  and ./out/kpi_wide.csv

CYCLE SUFFIXES: 2013-14 = _H, 2015-16 = _I, 2017-18 = _J, 2021-23 = _L
Variable names DO shift between cycles. Verify every name in the CDC codebook
for YOUR cycle before trusting a number. Names below are for 2017-2018 (_J).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DATA = Path("data")
OUT = Path("out")
OUT.mkdir(exist_ok=True)

CYCLE = "J"          # 2017-2018
N_CYCLES_POOLED = 1  # if you pool 2 cycles, set to 2 and see note in weight()
RATE_BASE = 100_000  # per 100,000 population


# ----------------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------------
def load(stem: str) -> pd.DataFrame:
    """Read one NHANES SAS transport file. Returns empty frame if missing."""
    path = DATA / f"{stem}_{CYCLE}.XPT"
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return pd.DataFrame()
    df = pd.read_sas(path, format="xport")
    df.columns = [c.upper() for c in df.columns]
    print(f"  [ok]   {path.name:16s} {len(df):>6,} rows  {len(df.columns):>3} cols")
    return df


def build_analytic_file() -> pd.DataFrame:
    print("Loading NHANES components...")
    demo = load("DEMO")            # demographics + weights + design vars
    if demo.empty:
        raise SystemExit("DEMO file is mandatory. Download it first.")

    frames = {
        "BMX": load("BMX"),        # body measures  -> BMXBMI, BMXWAIST
        "GHB": load("GHB"),        # glycohemoglobin -> LBXGH
        "GLU": load("GLU"),        # fasting glucose -> LBXGLU
        "TCHOL": load("TCHOL"),    # total cholesterol -> LBXTC
        "HDL": load("HDL"),        # HDL -> LBDHDD
        "BPXO": load("BPXO"),      # oscillometric BP (2017+) -> BPXOSY1..3
        "BPX": load("BPX"),        # manual BP (<=2015-16) -> BPXSY1..4
        "DIQ": load("DIQ"),        # diabetes questionnaire -> DIQ010
        "BPQ": load("BPQ"),        # BP/chol questionnaire -> BPQ020, BPQ080
        "MCQ": load("MCQ"),        # medical conditions -> MCQ160B/C/F/P
        "HUQ": load("HUQ"),        # hospital utilisation & access
        "HIQ": load("HIQ"),        # insurance -> HIQ011
        "SMQ": load("SMQ"),        # smoking -> SMQ020, SMQ040
        "DPQ": load("DPQ"),        # PHQ-9 depression -> DPQ010..DPQ090
        "RXQ_RX": load("RXQ_RX"),  # prescriptions -> RXDCOUNT
    }

    df = demo.copy()
    for name, f in frames.items():
        if f.empty:
            continue
        if name == "RXQ_RX":
            # RXQ_RX carries a placeholder row for non-users (RXDUSE == 2).
            # Count only rows representing an actual medication.
            if "RXDUSE" in f.columns:
                f = f[f["RXDUSE"] == 1]
            f = (f.groupby("SEQN", as_index=False)
                  .agg(RX_N=("SEQN", "size")))
        else:
            f = f.drop_duplicates("SEQN")
        df = df.merge(f, on="SEQN", how="left", suffixes=("", f"_{name}"))
    print(f"Analytic file: {len(df):,} persons\n")
    return df


# ----------------------------------------------------------------------------
# 2. DERIVE INDICATORS
# ----------------------------------------------------------------------------
def yes(s: pd.Series) -> pd.Series:
    """NHANES coding: 1=Yes, 2=No, 7=Refused, 9=Don't know, .=Missing.
    Returns 1/0 with 7/9/missing set to NaN so they drop out of the denominator."""
    return s.where(s.isin([1, 2])).map({1.0: 1.0, 2.0: 0.0})


def derive(df: pd.DataFrame) -> pd.DataFrame:
    d = df

    # --- design + demographics ------------------------------------------------
    d["AGE"] = d["RIDAGEYR"]
    d["SEX"] = d["RIAGENDR"].map({1.0: "Male", 2.0: "Female"})
    d["AGE_BAND"] = pd.cut(d["AGE"],
                           bins=[-1, 17, 29, 39, 49, 59, 69, 200],
                           labels=["0-17", "18-29", "30-39", "40-49",
                                   "50-59", "60-69", "70+"])
    # Separate age grouping for direct standardisation -- aligned to the NCHS
    # 2000 standard-population decade boundaries (15-24/25-34/.../85+), which do
    # NOT line up with AGE_BAND above. Kept distinct rather than reusing AGE_BAND.
    d["STD_AGE_GRP"] = pd.cut(d["AGE"],
                              bins=[-1, 24, 34, 44, 54, 64, 74, 84, 200],
                              labels=["18-24", "25-34", "35-44", "45-54",
                                      "55-64", "65-74", "75-84", "85+"])
    d["RACE_ETH"] = d["RIDRETH3"].map({
        1.0: "Mexican American", 2.0: "Other Hispanic",
        3.0: "Non-Hispanic White", 4.0: "Non-Hispanic Black",
        6.0: "Non-Hispanic Asian", 7.0: "Other/Multi"})
    # INDFMPIR = family income-to-poverty ratio. Quintiles are the equity axis
    # that replaces the (unavailable) state/county axis.
    if "INDFMPIR" in d:
        d["PIR_Q"] = pd.qcut(d["INDFMPIR"], 5, labels=[f"Q{i}" for i in range(1, 6)],
                             duplicates="drop")

    # --- Domain 2: chronic disease surveillance -------------------------------
    if "LBXGH" in d:
        d["DM_LAB"] = (d["LBXGH"] >= 6.5).where(d["LBXGH"].notna()).astype(float)
    if "DIQ010" in d:
        d["DM_DX"] = yes(d["DIQ010"])
    if {"DM_LAB", "DM_DX"} <= set(d.columns):
        d["DM_ANY"] = ((d["DM_LAB"] == 1) | (d["DM_DX"] == 1)).astype(float)
        d.loc[d["DM_LAB"].isna() & d["DM_DX"].isna(), "DM_ANY"] = np.nan
        # UNDIAGNOSED: lab-positive but never told they had it. This is NHANES's
        # signature contribution -- no claims or registry dataset can produce it.
        d["DM_UNDIAG"] = np.where((d["DM_LAB"] == 1) & (d["DM_DX"] == 0), 1.0,
                          np.where(d["DM_LAB"] == 1, 0.0, np.nan))
    if "LBXGH" in d:
        d["PREDM_LAB"] = d["LBXGH"].between(5.7, 6.4).where(d["LBXGH"].notna()).astype(float)

    # measured BP: mean of available readings, then JNC/ACC threshold
    sys_cols = [c for c in ["BPXOSY1", "BPXOSY2", "BPXOSY3",
                            "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if c in d]
    dia_cols = [c for c in ["BPXODI1", "BPXODI2", "BPXODI3",
                            "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"] if c in d]
    if sys_cols and dia_cols:
        d["SBP"] = d[sys_cols].replace(0, np.nan).mean(axis=1)
        d["DBP"] = d[dia_cols].replace(0, np.nan).mean(axis=1)
        d["HTN_LAB"] = ((d["SBP"] >= 130) | (d["DBP"] >= 80)).where(
            d["SBP"].notna() | d["DBP"].notna()).astype(float)
    if "BPQ020" in d:
        d["HTN_DX"] = yes(d["BPQ020"])
    if {"HTN_LAB", "HTN_DX"} <= set(d.columns):
        d["HTN_ANY"] = ((d["HTN_LAB"] == 1) | (d["HTN_DX"] == 1)).astype(float)
        d["HTN_UNDIAG"] = np.where((d["HTN_LAB"] == 1) & (d["HTN_DX"] == 0), 1.0,
                           np.where(d["HTN_LAB"] == 1, 0.0, np.nan))

    # Guideline-discordance check: BPQ020 was answered against clinical practice
    # keyed to 140/90, not the 2017 ACC/AHA 130/80 threshold. Compute both so the
    # gap between them -- not either number alone -- is the reportable finding.
    if "SBP" in d and "DBP" in d and "HTN_DX" in d:
        for lo_s, lo_d, tag in [(130, 80, "130"), (140, 90, "140")]:
            lab = ((d["SBP"] >= lo_s) | (d["DBP"] >= lo_d)).where(
                d["SBP"].notna() | d["DBP"].notna()).astype(float)
            d[f"HTN_LAB_{tag}"] = lab
            d[f"HTN_UNDIAG_{tag}"] = np.where((lab == 1) & (d["HTN_DX"] == 0), 1.0,
                                      np.where(lab == 1, 0.0, np.nan))

        # Reclassified band only (130-139/80-89, below the pre-2017 threshold).
        # Subtracting the two rates above is invalid -- different denominators.
        band = (d["HTN_LAB_130"] == 1) & (d["HTN_LAB_140"] == 0)
        d["HTN_UNDIAG_BAND"] = np.where(band & (d["HTN_DX"] == 0), 1.0,
                                np.where(band, 0.0, np.nan))

    if "BMXBMI" in d:
        d["OBESE"] = (d["BMXBMI"] >= 30).where(d["BMXBMI"].notna()).astype(float)
        d["SEV_OBESE"] = (d["BMXBMI"] >= 40).where(d["BMXBMI"].notna()).astype(float)

    # NOTE: MCQ160P does not exist in the 2017-2018 (_J) codebook. The direct
    # "ever told you had COPD" item in this cycle is MCQ160O (confirmed against
    # the CDC MCQ_J codebook) -- MCQ160G (emphysema) and MCQ160K (chronic
    # bronchitis) are separate, older-style items used pre-2017 to proxy COPD.
    for var, name in [("MCQ160B", "HF"), ("MCQ160C", "CHD"),
                      ("MCQ160F", "STROKE"), ("MCQ160O", "COPD")]:
        if var in d:
            d[name] = yes(d[var])

    # PHQ-9: items coded 0-3, 7/9 = refused/DK. Score >=10 = moderate+ depression.
    dpq = [f"DPQ0{i}0" for i in range(1, 10) if f"DPQ0{i}0" in d]
    if len(dpq) == 9:
        p = d[dpq].where(d[dpq] <= 3)
        d["PHQ9"] = p.sum(axis=1).where(p.notna().sum(axis=1) == 9)
        d["DEPRESSION"] = (d["PHQ9"] >= 10).where(d["PHQ9"].notna()).astype(float)

    # multimorbidity count -> drives the Level-3 heatmap
    cc = [c for c in ["DM_ANY", "HTN_ANY", "OBESE", "HF", "CHD", "COPD", "DEPRESSION"]
          if c in d]
    if cc:
        d["CC_COUNT"] = d[cc].fillna(0).sum(axis=1)
        d["CC_BAND"] = pd.cut(d["CC_COUNT"], [-1, 0, 1, 2, 99],
                              labels=["0", "1", "2", "3+"])

    # --- Domain 1: utilisation & patient experience ---------------------------
    if "HUQ010" in d:  # 1=Excellent ... 5=Poor
        d["HEALTH_FAIRPOOR"] = d["HUQ010"].isin([4, 5]).where(
            d["HUQ010"] <= 5).astype(float)
    if "HUQ071" in d:  # overnight hospital inpatient in past 12 months
        d["ADMITTED_12M"] = yes(d["HUQ071"])
    if "HUQ051" in d:  # number of times received healthcare in past 12 months
        d["HIGH_UTIL"] = (d["HUQ051"] >= 4).where(d["HUQ051"] <= 8).astype(float)
    if "HUQ030" in d:  # has a routine place to go for healthcare
        d["HAS_USUAL_CARE"] = yes(d["HUQ030"])
    if "HUQ041" in d:  # type of place most often used; 5 = hospital ED
        d["USUAL_CARE_IS_ED"] = (d["HUQ041"] == 5).where(d["HUQ041"] <= 6).astype(float)

    # --- Domain 3: access & demand-side capacity ------------------------------
    if "HIQ011" in d:
        d["INSURED"] = yes(d["HIQ011"])
        d["UNINSURED"] = 1 - d["INSURED"]
    if "RX_N" in d:
        d["RX_N"] = d["RX_N"].fillna(0)
        d["POLYPHARMACY"] = (d["RX_N"] >= 5).astype(float)

    if "SMQ020" in d and "SMQ040" in d:
        ever = yes(d["SMQ020"])
        d["SMOKER_CUR"] = np.where(d["SMQ040"].isin([1, 2]), 1.0,
                            np.where(ever == 0, 0.0,
                              np.where(d["SMQ040"] == 3, 0.0, np.nan)))
    return d


# ----------------------------------------------------------------------------
# 3. SURVEY-WEIGHTED ESTIMATION  (the part that makes it defensible)
# ----------------------------------------------------------------------------
def weight(d: pd.DataFrame, indicator: str) -> str:
    """MEC-examined weight for lab/exam variables, interview weight otherwise.
    POOLING: NHANES 2-year weights must be divided by the number of pooled
    cycles (e.g. WTMEC2YR/2 for two cycles). Do that here, not downstream."""
    exam_based = {"BMXBMI", "OBESE", "SEV_OBESE", "SEV_OBESE", "SBP", "DBP",
                  "HTN_LAB", "HTN_ANY", "HTN_UNDIAG", "DM_LAB", "DM_ANY",
                  "DM_UNDIAG", "PREDM_LAB", "CC_COUNT", "SMOKER_CUR",
                  "HTN_LAB_130", "HTN_UNDIAG_130", "HTN_LAB_140", "HTN_UNDIAG_140",
                  "HTN_UNDIAG_BAND"}
    return "WTMEC2YR" if indicator in exam_based else "WTINT2YR"


def taylor_proportion(sub: pd.DataFrame, y: str, w: str,
                      strata: str = "SDMVSTRA", psu: str = "SDMVPSU"):
    """
    Weighted proportion with Taylor-linearised SE for a with-replacement
    stratified cluster design -- the estimator NCHS itself recommends.

        p = sum(w*y) / sum(w)
        z_hi = sum over PSU i in stratum h of  w*(y - p)
        var(p) = (1/X^2) * sum_h [ n_h/(n_h-1) * sum_i (z_hi - zbar_h)^2 ]
    """
    s = sub[[y, w, strata, psu]].dropna()
    s = s[s[w] > 0]
    if s.empty:
        return dict(n=0, n_pos=0, p=np.nan, se=np.nan, lo=np.nan, hi=np.nan,
                    rse=np.nan, wpop=np.nan, suppress=True, dof=np.nan)

    X = s[w].sum()
    Y = (s[w] * s[y]).sum()
    p = Y / X

    s = s.assign(_r=s[w] * (s[y] - p))
    z = s.groupby([strata, psu], observed=True)["_r"].sum().reset_index()
    var = 0.0
    for _, g in z.groupby(strata, observed=True):
        n_h = len(g)
        if n_h < 2:            # lone PSU: contributes 0 (centre-at-grand-mean
            continue           # is the alternative; document whichever you use)
        var += n_h / (n_h - 1) * ((g["_r"] - g["_r"].mean()) ** 2).sum()
    var /= X ** 2
    se = float(np.sqrt(max(var, 0)))

    # Design degrees of freedom = PSUs - strata, NOT infinite. With ~15 strata x
    # 2 PSUs (NHANES 2017-18), dof ~= 15 -- t(0.975, 15) ~= 2.131, not 1.96.
    dof = max(len(z) - z[strata].nunique(), 1)
    tcrit = float(stats.t.ppf(0.975, dof))

    # logit-transformed CI: keeps bounds inside [0,1] for small proportions.
    # NCHS prefers Korn-Graubard; note the difference in your limitations section.
    if 0 < p < 1 and se > 0:
        lg = np.log(p / (1 - p))
        se_lg = se / (p * (1 - p))
        lo, hi = [1 / (1 + np.exp(-(lg + k * tcrit * se_lg))) for k in (-1, 1)]
    else:
        lo, hi = np.nan, np.nan

    n_eff = len(s)
    rse = (se / p * 100) if p > 0 else np.nan
    # NCHS Data Presentation Standards for Proportions: suppress unreliable cells
    suppress = bool(n_eff < 30 or (pd.notna(rse) and rse > 30))

    return dict(n=n_eff, n_pos=int(s[y].sum()), p=float(p), se=se,
                lo=lo, hi=hi, rse=rse, wpop=float(X), suppress=suppress, dof=dof)


def estimate(d: pd.DataFrame, indicator: str, by=None,
             universe: str = "adults") -> pd.DataFrame:
    """Estimate one indicator overall and/or across a stratification."""
    if indicator not in d.columns:
        return pd.DataFrame()
    w = weight(d, indicator)
    if w not in d.columns:
        return pd.DataFrame()

    base = d if universe == "all" else d[d["AGE"] >= 18]
    by = by or []
    rows = []
    groups = [((), base)] if not by else list(base.groupby(by, observed=True))
    for key, g in groups:
        r = taylor_proportion(g, indicator, w)
        keyt = key if isinstance(key, tuple) else (key,)
        rows.append({
            "indicator": indicator,
            "stratifier": " x ".join(by) if by else "Overall",
            "stratum": " | ".join(map(str, keyt)) if by else "Overall",
            "percent": None if r["suppress"] else round(r["p"] * 100, 2),
            "rate_per_100k": None if r["suppress"] else round(r["p"] * RATE_BASE, 1),
            "ci_lo_per_100k": None if r["suppress"] else round(r["lo"] * RATE_BASE, 1),
            "ci_hi_per_100k": None if r["suppress"] else round(r["hi"] * RATE_BASE, 1),
            "se_pct": round(r["se"] * 100, 3),
            "rse_pct": None if pd.isna(r["rse"]) else round(r["rse"], 1),
            "unweighted_n": r["n"],
            "weighted_pop": None if pd.isna(r["wpop"]) else round(r["wpop"]),
            "suppressed": r["suppress"],
            "weight_var": w,
            "dof": None if pd.isna(r["dof"]) else int(r["dof"]),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 4. EQUITY: concentration index (replaces "regional concentration index")
# ----------------------------------------------------------------------------
def concentration_index(d: pd.DataFrame, indicator: str,
                        rank_var: str = "INDFMPIR", min_age: int = 18) -> float:
    """
    Erreygers-style weighted concentration index over the income-to-poverty
    ratio. CI < 0 => the condition is concentrated among the poor.
    This is a real health-economics metric and a legitimate substitute for the
    geographic concentration index that NHANES public files cannot support.

    Restricted to AGE >= min_age -- BMXBMI exists from age 2 and LBXGH from
    age 12, so leaving children in pulls disease indices toward the top of
    the income distribution (kids are mostly disease-free and skew low-PIR).

    INDFMPIR is top-coded at 5.00, so ~1/6 of respondents sit on one tied
    value. Assign every tied value the weighted midpoint of its block instead
    of letting a naive cumulative rank spread them across an arbitrary range.
    """
    if indicator not in d or rank_var not in d:
        return np.nan
    w = weight(d, indicator)
    s = d.loc[d["AGE"] >= min_age, [indicator, rank_var, w]].dropna()
    s = s[s[w] > 0].sort_values(rank_var)
    if len(s) < 100:
        return np.nan
    wt = s[w].to_numpy()
    y = s[indicator].to_numpy()
    p = pd.Series(wt / wt.sum())
    cw = pd.Series(np.cumsum(wt) / wt.sum())
    grp = pd.Series(s[rank_var].to_numpy())
    f_hi = cw.groupby(grp).transform("max")
    f_lo = f_hi - p.groupby(grp).transform("sum")
    frank = ((f_lo + f_hi) / 2).to_numpy()
    mu = np.average(y, weights=wt)
    if mu == 0:
        return np.nan
    return float(2 / mu * np.average((y - mu) * (frank - 0.5), weights=wt))


# ----------------------------------------------------------------------------
# 4b. AGE STANDARDISATION (direct method, 2000 U.S. standard population)
# ----------------------------------------------------------------------------
# Source: Census-Bureau-projected year-2000 U.S. standard population, as used
# by NCHS for age adjustment (Klein RJ, Schoenborn CA. Age adjustment using the
# 2000 projected U.S. population. Healthy People 2000 Stat Notes 2001;20;
# reproduced in Health, United States, 2019, Appendix II Table 2). Standard
# 11-group weights, "standard million":
#   <1: 13818  1-4: 55317  5-14: 145565  15-24: 138646  25-34: 135573
#   35-44: 162613  45-54: 134834  55-64: 87247  65-74: 66037  75-84: 44842
#   85+: 15508
#
# DISCLOSED LIMITATION: the youngest official bracket is 15-24, but this
# analysis is adults-only (18+). The 15-24 weight is split into 18-24 and
# 15-17 by assuming a UNIFORM population distribution across the ten single
# ages in that decade (no single-year-of-age table was available to verify
# this split) -- 18-24 is then 7/10 of the 15-24 weight. This is an explicit,
# disclosed approximation, not a fitted or invented number; state it in
# Methodology and Limitations.
_STD_WEIGHTS_RAW = {
    "18-24": 138646 * 7 / 10,   # approximation -- see note above
    "25-34": 135573,
    "35-44": 162613,
    "45-54": 134834,
    "55-64": 87247,
    "65-74": 66037,
    "75-84": 44842,
    "85+": 15508,
}
STD_WEIGHTS = {k: v / sum(_STD_WEIGHTS_RAW.values()) for k, v in _STD_WEIGHTS_RAW.items()}


def age_standardize(d: pd.DataFrame, indicator: str) -> dict:
    """
    Direct age standardisation to the 2000 U.S. standard population (adults
    18+ subset, weights renormalised to sum to 1 over that subset -- the
    approach Klein & Schoenborn describe for a restricted analytic population).

    std_rate = sum_g( std_weight_g * p_g ) / sum_g( std_weight_g used )
    std_var  = sum_g( std_weight_g^2 * var_g ) / (sum_g std_weight_g used)^2

    Age groups suppressed by the same RSE>30%/n<30 rule used elsewhere are
    dropped and the remaining weights renormalised; n_groups_used / 8 tells you
    how much of the standard population that estimate actually rests on.
    A group entirely absent from this cell (e.g. no 85+ respondents in a small
    stratum) is dropped the same way.

    CI uses a normal (z=1.96) logit-transformed interval, NOT the t(dof=15)
    interval used for single-cell crude estimates: the age-standardised
    estimate pools across STD_AGE_GRP cells that don't share one clean design
    degrees-of-freedom count, so there is no single defensible dof to plug in
    here. This is a documented simplification, consistent with common NCHS
    practice for standardised rates; note it in Methodology.
    """
    if indicator not in d.columns:
        return dict(std_p=np.nan, std_lo=np.nan, std_hi=np.nan,
                    n_groups_used=0, n_total=len(STD_WEIGHTS), suppressed=True)
    w = weight(d, indicator)
    if w not in d.columns or "STD_AGE_GRP" not in d.columns:
        return dict(std_p=np.nan, std_lo=np.nan, std_hi=np.nan,
                    n_groups_used=0, n_total=len(STD_WEIGHTS), suppressed=True)

    # Caller is responsible for pre-filtering to the stratum of interest
    # (e.g. SEX=='Male'); AGE>=18 is enforced here regardless as a floor.
    base = d[d["AGE"] >= 18]

    used_w, used_p, used_var = [], [], []
    for grp, wt_g in STD_WEIGHTS.items():
        cell = base[base["STD_AGE_GRP"] == grp]
        r = taylor_proportion(cell, indicator, w)
        if r["suppress"] or pd.isna(r["p"]):
            continue
        used_w.append(wt_g)
        used_p.append(r["p"])
        used_var.append(r["se"] ** 2)

    n_used = len(used_w)
    if n_used == 0:
        return dict(std_p=np.nan, std_lo=np.nan, std_hi=np.nan,
                    n_groups_used=0, n_total=len(STD_WEIGHTS), suppressed=True)

    used_w = np.array(used_w)
    wsum = used_w.sum()
    std_p = float(np.sum(used_w * np.array(used_p)) / wsum)
    std_var = float(np.sum(used_w ** 2 * np.array(used_var)) / wsum ** 2)
    std_se = float(np.sqrt(max(std_var, 0)))

    if 0 < std_p < 1 and std_se > 0:
        lg = np.log(std_p / (1 - std_p))
        se_lg = std_se / (std_p * (1 - std_p))
        std_lo, std_hi = [1 / (1 + np.exp(-(lg + k * 1.96 * se_lg))) for k in (-1, 1)]
    else:
        std_lo, std_hi = np.nan, np.nan

    return dict(std_p=std_p, std_se=std_se, std_lo=std_lo, std_hi=std_hi,
                n_groups_used=n_used, n_total=len(STD_WEIGHTS),
                suppressed=(n_used < len(STD_WEIGHTS) // 2))


# ----------------------------------------------------------------------------
# 5. RUN
# ----------------------------------------------------------------------------
DASHBOARD_SPEC = {
    # Domain 1 -- Utilisation & Patient Experience
    "L1_util":   ["HEALTH_FAIRPOOR", "ADMITTED_12M", "HIGH_UTIL"],
    "L2_util":   ["ADMITTED_12M", "HAS_USUAL_CARE", "USUAL_CARE_IS_ED"],
    # Domain 2 -- Chronic Disease Surveillance
    "L1_dis":    ["DM_ANY", "HTN_ANY", "OBESE"],
    "L2_dis":    ["DM_UNDIAG", "HTN_UNDIAG", "HTN_UNDIAG_130", "HTN_UNDIAG_140",
                  "HTN_UNDIAG_BAND", "PREDM_LAB", "DEPRESSION"],
    # Domain 3 -- Access & Demand-side Capacity
    "L1_cap":    ["UNINSURED", "HAS_USUAL_CARE", "POLYPHARMACY"],
}

STRATIFIERS = [
    [],                          # executive tier: single headline number
    ["AGE_BAND"],
    ["SEX"],
    ["AGE_BAND", "SEX"],         # feeds the population pyramid
    ["RACE_ETH"],
    ["PIR_Q"],                   # equity gradient -> the "allocation gap"
    ["CC_BAND"],                 # multimorbidity -> feeds the heatmap
]


def main():
    d = derive(build_analytic_file())

    indicators = sorted({i for v in DASHBOARD_SPEC.values() for i in v})
    print("Estimating (survey-weighted, Taylor SE)...")
    out = []
    for ind in indicators:
        if ind not in d.columns:
            print(f"  [miss] {ind} -- source file not downloaded")
            continue
        for by in STRATIFIERS:
            if any(b not in d.columns for b in by):
                continue
            out.append(estimate(d, ind, by))
        print(f"  [done] {ind}")

    if not out:
        raise SystemExit("Nothing estimated. Check that data/ contains .XPT files.")

    long = pd.concat(out, ignore_index=True)
    long["cycle"] = {"H": "2013-2014", "I": "2015-2016",
                     "J": "2017-2018", "L": "2021-2023"}.get(CYCLE, CYCLE)
    long.to_csv(OUT / "kpi_long.csv", index=False)

    (long[long.stratifier == "Overall"]
        .pivot_table(index="cycle", columns="indicator", values="rate_per_100k")
        .to_csv(OUT / "kpi_wide.csv"))

    ci = pd.DataFrame([{"indicator": i,
                        "concentration_index_income": round(concentration_index(d, i), 4)}
                       for i in indicators if i in d.columns])
    ci.to_csv(OUT / "concentration_index.csv", index=False)

    print(f"\nWrote {OUT/'kpi_long.csv'}  ({len(long):,} rows)")
    print(f"Wrote {OUT/'kpi_wide.csv'}")
    print(f"Wrote {OUT/'concentration_index.csv'}")
    print("\nSuppressed cells (RSE>30% or n<30):",
          int(long["suppressed"].sum()), "of", len(long))
    print("\nNEXT: point Power BI / Tableau / Streamlit at out/kpi_long.csv.")
    print("For the choropleth you MUST join CDC PLACES (state/county) -- NHANES")
    print("public-use files contain no geographic identifiers.")


if __name__ == "__main__":
    main()
