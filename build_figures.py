"""
Task 2b: six figures, 300dpi PNG + SVG, each with a caption stating
source / cycle / weight / denominator / n directly on the figure.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

CYCLE = "NHANES 2017-2018 (_J)"
OUTDIR = "figures"


def save(fig, name):
    fig.savefig(f"{OUTDIR}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUTDIR}/{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {name}.png / .svg")


def add_caption(fig, text):
    fig.text(0.02, -0.02, text, ha="left", va="top", fontsize=7.5, wrap=True,
              transform=fig.transFigure, color="#333333")


# ---------------------------------------------------------------------------
# 1. Scatter: HIGH_UTIL vs HEALTH_FAIRPOOR, bubble = weighted pop
# ---------------------------------------------------------------------------
sc = pd.read_csv("out/fig_scatter_util_vs_health.csv").dropna(subset=["high_util_pct", "health_fairpoor_pct"])
fig, ax = plt.subplots(figsize=(7, 6))
colors = {"Male": "#2b6cb0", "Female": "#c05621"}
for sex, g in sc.groupby("sex"):
    sizes = g["weighted_pop"] / g["weighted_pop"].max() * 2500 + 100
    ax.scatter(g["high_util_pct"], g["health_fairpoor_pct"], s=sizes,
               alpha=0.6, color=colors[sex], label=sex, edgecolor="white", linewidth=0.8)
    for _, row in g.iterrows():
        ax.annotate(row["age_band"], (row["high_util_pct"], row["health_fairpoor_pct"]),
                    fontsize=7, ha="center", va="center")
ax.set_xlabel("High healthcare utilisation, >=4 contacts/12mo (%)")
ax.set_ylabel("Fair/poor self-rated health (%)")
ax.set_title("Utilisation vs. self-rated health, by age band and sex")
ax.legend(title="Sex", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
ax.grid(alpha=0.3)
ax.margins(0.15)
add_caption(fig, f"Source: {CYCLE}, weight=WTINT2YR. Denominator: adults 18+ within each age-band x sex cell "
                  f"(n per cell in out/fig_scatter_util_vs_health.csv). Bubble area proportional to weighted "
                  f"population of the cell. HIGH_UTIL = HUQ051>=4; HEALTH_FAIRPOOR = HUQ010 in {{4,5}}.")
save(fig, "scatter_util_vs_health")

# ---------------------------------------------------------------------------
# 2. Heatmap: AGE_BAND x CC_BAND -> ADMITTED_12M rate/100k
# ---------------------------------------------------------------------------
heat = pd.read_csv("out/fig_heatmap_admission_by_age_cc.csv")
age_order = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]
cc_order = ["0", "1", "2", "3+"]
mat = heat.pivot(index="age_band", columns="cc_band", values="rate_per_100k").reindex(index=age_order, columns=cc_order)
n_mat = heat.pivot(index="age_band", columns="cc_band", values="n").reindex(index=age_order, columns=cc_order)

fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(mat.values.astype(float), cmap="OrRd", aspect="auto")
ax.set_xticks(range(len(cc_order))); ax.set_xticklabels(cc_order)
ax.set_yticks(range(len(age_order))); ax.set_yticklabels(age_order)
ax.set_xlabel("Chronic condition count (CC_BAND)")
ax.set_ylabel("Age band")
ax.set_title("12-month hospital admission rate per 100,000, by age and multimorbidity")
for i in range(len(age_order)):
    for j in range(len(cc_order)):
        val = mat.values[i, j]
        n_val = n_mat.values[i, j]
        if pd.isna(val):
            ax.text(j, i, f"suppressed\n(n={n_val})", ha="center", va="center", fontsize=6.5, color="#666")
        else:
            ax.text(j, i, f"{val:,.0f}\n(n={n_val})", ha="center", va="center", fontsize=7,
                     color="white" if val > np.nanmax(mat.values.astype(float)) * 0.5 else "black")
cbar = fig.colorbar(im, ax=ax); cbar.set_label("Rate per 100,000")
add_caption(fig, f"Source: {CYCLE}, weight=WTINT2YR. ADMITTED_12M = overnight hospital stay in past 12 months "
                  f"(HUQ071). Denominator: adults 18+ within each age-band x CC_BAND cell; n shown per cell. "
                  f"Cells with n<30 or RSE>30% suppressed per NCHS Data Presentation Standards.")
save(fig, "heatmap_admission_age_cc")

# ---------------------------------------------------------------------------
# 3. Population pyramid (rate-based): HTN_ANY by age band x sex
# ---------------------------------------------------------------------------
pyr = pd.read_csv("out/fig_pyramid_htn_by_age_sex.csv")
male = pyr[pyr.sex == "Male"].set_index("age_band").reindex(age_order)
female = pyr[pyr.sex == "Female"].set_index("age_band").reindex(age_order)

fig, ax = plt.subplots(figsize=(7, 5.5))
y = np.arange(len(age_order))
ax.barh(y, -male["rate_pct"].fillna(0), color="#2b6cb0", label="Male")
ax.barh(y, female["rate_pct"].fillna(0), color="#c05621", label="Female")
for i, (m, f) in enumerate(zip(male["rate_pct"], female["rate_pct"])):
    if pd.notna(m):
        ax.text(-m - 1, i, f"{m:.0f}%", va="center", ha="right", fontsize=8)
    if pd.notna(f):
        ax.text(f + 1, i, f"{f:.0f}%", va="center", ha="left", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(age_order)
ax.set_xlabel("Hypertension prevalence (%) <- Male   |   Female ->")
ax.set_title("Rate-based population pyramid: hypertension prevalence by age and sex")
ax.axvline(0, color="black", linewidth=0.8)
xt = ax.get_xticks(); ax.set_xticks(xt); ax.set_xticklabels([f"{abs(int(t))}%" for t in xt])
ax.legend()
add_caption(fig, f"Source: {CYCLE}, weight=WTMEC2YR. HTN_ANY = measured BP>=130/80 (BPXO) OR self-report "
                  f"(BPQ020), 2017 ACC/AHA threshold. Denominator: adults 18+ within each age-band x sex cell "
                  f"(n in out/fig_pyramid_htn_by_age_sex.csv). This is a RATE pyramid (prevalence), not a "
                  f"population-count pyramid.")
save(fig, "pyramid_htn_rate_by_age_sex")

# ---------------------------------------------------------------------------
# 4. Bar-in-bar: actual vs Healthy People 2030 targets
# ---------------------------------------------------------------------------
hp = pd.DataFrame([
    {"indicator": "HTN_ANY", "mine_std": 45.00, "hp2030_baseline": 45.0, "hp2030_target": 41.9,
     "objective": "HDS-04", "note": "baseline 2017-2020, NHANES, 18+"},
    {"indicator": "OBESE", "mine_std": 41.80, "hp2030_baseline": 38.6, "hp2030_target": 36.0,
     "objective": "NWS-03", "note": "baseline 2013-2016, NHANES, 20+"},
    {"indicator": "UNINSURED", "mine_std": 15.58, "hp2030_baseline": 12.0, "hp2030_target": 7.6,
     "objective": "AHS-01 (inverted from insured)", "note": "baseline 2019, NHIS, under 65"},
])
fig, ax = plt.subplots(figsize=(7.5, 5.5))
x = np.arange(len(hp))
w = 0.25
ax.bar(x - w, hp["mine_std"], width=w, label="This study (age-std.)", color="#2b6cb0")
ax.bar(x, hp["hp2030_baseline"], width=w, label="HP2030 baseline", color="#a0aec0")
ax.bar(x + w, hp["hp2030_target"], width=w, label="HP2030 target", color="#38a169")
ax.set_xticks(x); ax.set_xticklabels(hp["indicator"])
ax.set_ylabel("Percent")
ax.set_title("This study vs. Healthy People 2030 baseline and target")
for i, row in hp.iterrows():
    ax.annotate(row["objective"], (i, max(row["mine_std"], row["hp2030_baseline"]) + 2),
                ha="center", fontsize=7)
ax.legend()
add_caption(fig, f"Source: {CYCLE} (this study, weight=WTMEC2YR for HTN_ANY/OBESE, WTINT2YR for UNINSURED). "
                  f"HP2030 targets fetched from odphp.health.gov (HDS-04, NWS-03, AHS-01) -- see T7/references "
                  f"for exact baseline periods and age ranges, which differ from this study's (noted per bar). "
                  f"No HP2030 objective measures diabetes PREVALENCE (D-01 measures incidence, a different "
                  f"quantity) -- diabetes is deliberately omitted here rather than force-matched.")
save(fig, "barinbar_vs_hp2030")

# ---------------------------------------------------------------------------
# 5. Stacked bar: Domain 3 access/capacity indicators by PIR quintile
# ---------------------------------------------------------------------------
acc = pd.read_csv("out/fig_stackedbar_access_by_pir.csv")
q_order = ["Q1", "Q2", "Q3", "Q4", "Q5"]
fig, ax = plt.subplots(figsize=(7.5, 5.5))
bottom = np.zeros(len(q_order))
colors3 = {"UNINSURED": "#c53030", "HAS_USUAL_CARE": "#2b6cb0", "POLYPHARMACY": "#805ad5"}
for ind in ["UNINSURED", "HAS_USUAL_CARE", "POLYPHARMACY"]:
    vals = acc[acc.indicator == ind].set_index("pir_q").reindex(q_order)["percent"].fillna(0).values
    ax.bar(q_order, vals, bottom=bottom, label=ind, color=colors3[ind])
    bottom += vals
ax.set_xlabel("Income-to-poverty ratio quintile (Q1=poorest)")
ax.set_ylabel("Percent (stacked, not mutually exclusive)")
ax.set_title("Access & capacity indicators by income quintile")
ax.legend()
add_caption(fig, f"Source: {CYCLE}, weight=WTINT2YR (all three indicators). Denominator: adults 18+ per "
                  f"PIR quintile (n in "
                  f"out/fig_stackedbar_access_by_pir.csv). NOTE: these three indicators are not mutually "
                  f"exclusive (a person can be both uninsured and on 5+ medications) -- bar heights are "
                  f"additive for visual comparison only, not a partition of the population.")
save(fig, "stackedbar_access_by_pir")

# ---------------------------------------------------------------------------
# 6. Treemap: burden share by condition (weighted population), no budget data
# ---------------------------------------------------------------------------
def squarify(sizes, x, y, w, h):
    """Minimal squarified treemap layout (Bruls et al. 1999), sizes pre-normalised to area w*h."""
    if not sizes:
        return []
    if len(sizes) == 1:
        return [(x, y, w, h)]
    total = sum(sizes)
    if w >= h:
        split = 0
        s = 0
        for i, sz in enumerate(sizes):
            s += sz
            if s >= total / 2:
                split = i + 1
                break
        split = max(1, min(split, len(sizes) - 1))
        left, right = sizes[:split], sizes[split:]
        lw = w * sum(left) / total
        return squarify(left, x, y, lw, h) + squarify(right, x + lw, y, w - lw, h)
    else:
        split = 0
        s = 0
        for i, sz in enumerate(sizes):
            s += sz
            if s >= total / 2:
                split = i + 1
                break
        split = max(1, min(split, len(sizes) - 1))
        top, bot = sizes[:split], sizes[split:]
        th = h * sum(top) / total
        return squarify(top, x, y, w, th) + squarify(bot, x, y + th, w, h - th)

burden = pd.read_csv("out/fig_treemap_burden.csv").dropna(subset=["weighted_pop_with_condition"])
burden = burden.sort_values("weighted_pop_with_condition", ascending=False)
sizes = burden["weighted_pop_with_condition"].tolist()
rects = squarify(sizes, 0, 0, 100, 100)

fig, ax = plt.subplots(figsize=(8, 6))
palette = plt.cm.tab10(np.linspace(0, 1, len(rects)))
for (x, y, w, h), (_, row), color in zip(rects, burden.iterrows(), palette):
    ax.add_patch(mpatches.Rectangle((x, y), w, h, facecolor=color, edgecolor="white", linewidth=2))
    if w > 8 and h > 8:
        ax.text(x + w/2, y + h/2, f"{row['condition']}\n{row['weighted_pop_with_condition']/1e6:.1f}M\n({row['percent']:.1f}%)",
                ha="center", va="center", fontsize=8.5, color="white", weight="bold")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.axis("off")
ax.set_title("Weighted population burden share by chronic condition (adults 18+)")
add_caption(fig, f"Source: {CYCLE}, mixed weights (WTMEC2YR for DM_ANY/HTN_ANY/OBESE, WTINT2YR for "
                  f"HF/CHD/COPD/DEPRESSION). Denominator: adults 18+, weight_var per condition. Area = "
                  f"weighted number of US adults with the condition (not mutually exclusive across "
                  f"conditions -- this is burden SHARE by condition, not a partition of one population). "
                  f"NO COST OR BUDGET DATA EXISTS IN NHANES -- this treemap shows population burden only, "
                  f"not resource allocation.")
save(fig, "treemap_burden_share")

print("\nAll 6 figures written to figures/ (PNG 300dpi + SVG).")
