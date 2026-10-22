"""Builds the 14-slide project presentation from the paper's figures and results."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

HERE = Path(__file__).parent
FIG = HERE / "figures"
OUT = HERE / "Varun_Mundhada_23070521184_NHANES_Project.pptx"
REPO = "https://github.com/varunmundhada/nhanes-healthcare-analytics"

NAVY, RED, GREY, LIGHT = RGBColor(0x1F, 0x4E, 0x79), RGBColor(0xC0, 0x39, 0x2B), RGBColor(0x55, 0x55, 0x55), RGBColor(0xEE, 0xF3, 0xFA)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide(title, n):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(1.0))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.18), Inches(12.3), Inches(0.7)).text_frame
    tb.text = title
    p = tb.paragraphs[0]; p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = RGBColor(255, 255, 255)
    ft = s.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(12.3), Inches(0.35)).text_frame
    ft.text = f"Varun Mundhada  |  PRN 23070521184  |  Healthcare Analytics  |  {n}"
    ft.paragraphs[0].font.size = Pt(11); ft.paragraphs[0].font.color.rgb = GREY
    return s


def bullets(s, items, x=0.6, y=1.3, w=12.1, h=5.6, size=20):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        sub = it.startswith("  ")
        p.text = ("    - " if sub else "• ") + it.strip()
        p.font.size = Pt(size - 3 if sub else size)
        p.space_after = Pt(8)
    return tf


def image(s, name, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return s.shapes.add_picture(str(FIG / name), Inches(x), Inches(y), **kw)


def table(s, rows, x, y, w, col_w, size=14):
    t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(0.4 * len(rows))).table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = t.cell(i, j); c.text = str(v)
            para = c.text_frame.paragraphs[0]; para.font.size = Pt(size)
            if i == 0:
                para.font.bold = True; para.font.color.rgb = RGBColor(255, 255, 255)
                c.fill.solid(); c.fill.fore_color.rgb = NAVY
            else:
                c.fill.solid(); c.fill.fore_color.rgb = LIGHT if i % 2 else RGBColor(255, 255, 255)
    return t


# 1 Title
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height); bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
tf = s.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(3)).text_frame; tf.word_wrap = True
tf.text = "Design-Based KPI Estimation and Survey-Weighted Machine Learning for Undiagnosed Diabetes Screening on NHANES 2017-2018"
tf.paragraphs[0].font.size = Pt(36); tf.paragraphs[0].font.bold = True; tf.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
for line, sz in [("Varun Mundhada  |  PRN 23070521184", 22), ("B.Tech CSE, Semester VII  |  Healthcare Analytics, CA3 Mini Project", 18),
                 ("Symbiosis Institute of Technology, Nagpur", 18), (f"Code: {REPO}", 16)]:
    p = tf.add_paragraph(); p.text = line; p.font.size = Pt(sz); p.font.color.rgb = RGBColor(0xDD, 0xE6, 0xF2); p.space_before = Pt(10)

# 2 Problem
s = slide("Problem: disease that has never been diagnosed", 2)
bullets(s, ["Hospital records, claims and registries only see people who reached care",
            "Someone with undetected diabetes or hypertension appears in none of them",
            "NHANES interviews and physically examines a national US sample, so it can measure undiagnosed disease",
            "Course brief: build a three-tier KPI dashboard (executive, tactical, operational) with rates per 100,000",
            "Our question: how large is the diagnosis gap, who does it affect, and can simple non-lab information find these people?"])

# 3 Objectives
s = slide("Objectives and contributions", 3)
bullets(s, ["Survey-weighted pipeline: 612 KPI estimates with valid 95% confidence intervals",
            "Income inequality with the concentration index (pooled and within age bands)",
            "Three-tier Streamlit dashboard across three domains",
            "  Utilisation and patient experience, chronic disease surveillance, access and capacity",
            "Screening models for undiagnosed diabetes: LR, RF, gradient boosting, MLP vs the ADA risk score",
            "Comparison with 7 published models; 26 references (20 from 2021 or later)"])

# 4 Dataset
s = slide("Dataset: NHANES 2017-2018 (CDC / NCHS)", 4)
bullets(s, ["11 public-use files: DEMO, GHB, BPXO, BMX, DIQ, BPQ, MCQ, HUQ, HIQ, DPQ, RXQ_RX",
            "Merged on SEQN: 9,254 people, no duplicates",
            "Stratified multistage sample: 15 strata, 30 PSUs, design degrees of freedom = 15",
            "Weights: WTMEC2YR for exam/lab indicators, WTINT2YR for interview-only indicators",
            "17 binary indicators for adults 18+; codes 7/9 (refused, don't know) set to missing",
            "No geography in public files, so the state map uses CDC PLACES (separate, model-based)"], w=7.4, size=18)
table(s, [["Domain", "Example indicators"],
          ["Utilisation", "Fair/poor health, admitted 12m, 4+ visits"],
          ["Chronic disease", "Diabetes (HbA1c), undiagnosed DM/HTN, obesity, PHQ-9"],
          ["Access", "Uninsured, usual source of care, polypharmacy"]], 8.2, 1.6, 4.7, [1.5, 3.2], size=13)

# 5 Architecture
s = slide("System architecture", 5)
image(s, "architecture.png", 0.7, 1.3, w=11.9)

# 6 Methodology
s = slide("Design-based estimation methodology", 6)
bullets(s, ["Weighted prevalence: p = sum(w y) / sum(w)",
            "Taylor-linearised variance over strata and PSUs (with-replacement approximation)",
            "Logit-scale 95% CI with t(0.975, 15) = 2.131 instead of 1.96",
            "Suppress a cell if RSE > 30% or n < 30 (NCHS standard): 53 of 612 cells suppressed",
            "Concentration index C = (2 / mu) cov_w(y, R), tie-corrected rank for top-coded income; Erreygers E = 4 mu C",
            "Direct age standardisation to the US 2000 standard population (8 adult groups)"])

# 7 Dashboard
s = slide("Three-tier dashboard (Streamlit)", 7)
image(s, "dashboard_pair.png", 0.5, 1.25, w=12.3)
tf = s.shapes.add_textbox(Inches(0.6), Inches(6.35), Inches(12), Inches(0.6)).text_frame
tf.text = "L1 tiles with CI, RSE and n  ->  L2 breakdowns by age, sex, race, income  ->  L3 analytical charts. Suppressed cells shown grey, never as zero."
tf.paragraphs[0].font.size = Pt(15); tf.paragraphs[0].font.color.rgb = GREY

# 8 KPI results
s = slide("Key results: the diagnosis gap", 8)
table(s, [["Indicator (adults 18+)", "%", "95% CI", "n"],
          ["Any diabetes", "13.4", "12.1-14.8", "5,843"],
          ["Undiagnosed diabetes (of HbA1c >= 6.5%)", "18.0", "14.8-21.7", "746"],
          ["Undiagnosed hypertension, 130/80", "49.9", "45.4-54.4", "2,059"],
          ["Obesity (BMI >= 30)", "42.0", "38.2-45.8", "5,434"],
          ["Uninsured", "13.9", "10.3-18.5", "5,837"],
          ["Fair/poor self-rated health", "18.6", "16.6-20.8", "5,849"]], 0.6, 1.4, 7.6, [4.3, 0.9, 1.5, 0.9], size=15)
bullets(s, ["About 4.2 million US adults had undiagnosed diabetes",
            "Borderline answers counted as undiagnosed: 23.4%",
            "Hypertension gap is inflated by single-visit BP, so diabetes anchors the finding"], x=8.5, y=1.4, w=4.4, size=17)

# 9 Equity
s = slide("Who is affected: income inequality", 9)
image(s, "stackedbar_access_by_pir.png", 0.5, 1.2, h=5.6)
bullets(s, ["Uninsured: C = -0.42 (E = -0.24), strongly concentrated among the poor",
            "Steeper within age bands (-0.39 to -0.53), so not a Medicare effect",
            "Undiagnosed diabetes: C = -0.01, flat",
            "Insurance closes the access gap, but screening must reach every income group"], x=7.3, y=1.4, w=5.6, size=17)

# 10 Validation
s = slide("Validation against published NCHS estimates", 10)
table(s, [["Indicator", "This study (age-std.)", "Published", "Source"],
          ["Obesity", "41.8%", "42.4%", "Hales et al., Data Brief 360"],
          ["Hypertension", "45.0%", "45.4%", "Ostchega et al., Data Brief 364"],
          ["Any diabetes", "13.5%", "13.2% (12.0-14.5)", "CDC National Diabetes Statistics Report"]], 0.6, 1.5, 12.1, [2.4, 2.6, 2.6, 4.5], size=16)
bullets(s, ["Hypertension match may hide offsetting definitions (theirs counts medication users)",
            "Diabetes not lower despite omitting fasting glucose: an open question we report, not explain away"], y=3.8, size=17)

# 11 ML setup
s = slide("Screening model: setup and hyperparameters", 11)
bullets(s, ["Cohort: adults 20+, never told diabetes, HbA1c known: n = 4,051, 150 cases (3.7%)",
            "Target: HbA1c >= 6.5% (undiagnosed diabetes); 18 non-lab features",
            "Nested CV: 5 outer folds, 3-fold inner grid search on weighted AUC; MEC weights in fit and evaluation",
            "Baseline: adapted ADA risk score (5 of 7 items, max 9, cut-off >= 5)"], size=17, h=2.6)
table(s, [["Model", "Search space"],
          ["Logistic regression", "C in {0.01, 0.1, 1, 10}, L2, balanced classes"],
          ["Random forest", "500 trees; depth in {4, 8, none}; min leaf in {5, 20}"],
          ["Gradient boosting", "lr in {0.03, 0.1}; depth in {2, 3}; L2 in {0, 1}"],
          ["MLP", "hidden (16) or (32,16); alpha in {0.001, 0.1}; Adam"]], 0.6, 4.1, 12.1, [3.0, 9.1], size=15)

# 12 ML results
s = slide("Screening results", 12)
image(s, "ml_roc.png", 0.4, 1.2, h=4.6)
image(s, "ml_importance.png", 5.6, 1.4, h=3.9)
table(s, [["Model", "Pooled weighted AUC (95% CI)", "Sens @ 80% spec"],
          ["Logistic regression", "0.788 (0.732-0.841)", "0.62"],
          ["Random forest", "0.759 (0.691-0.820)", "0.58"],
          ["Gradient boosting", "0.737 (0.673-0.804)", "0.53"],
          ["ADA score (adapted)", "0.734 (0.689-0.775)", "0.45"],
          ["MLP (unweighted)", "0.552 (0.474-0.628)", "0.21"]], 0.6, 5.0, 8.0, [2.6, 3.4, 2.0], size=12)
tf = s.shapes.add_textbox(Inches(8.9), Inches(5.0), Inches(4.1), Inches(1.9)).text_frame; tf.word_wrap = True
tf.text = "LR gain over ADA: +0.057 (95% CI -0.007 to +0.114), not significant with 150 cases"
tf.paragraphs[0].font.size = Pt(15); tf.paragraphs[0].font.color.rgb = RED

# 13 Comparison
s = slide("Comparison with published models", 13)
table(s, [["Study", "Data / outcome", "Best reported AUC"],
          ["De Silva 2021", "NHANES, undiagnosed T2D", "LR 0.757; ADA 0.737"],
          ["Choi 2023", "KNHANES, undiagnosed DM", "LightGBM 0.819"],
          ["Qin 2022", "NHANES, fasting glucose", "CatBoost 0.83"],
          ["Riveros Perez 2025", "NHANES, self-reported T2D", "XGBoost 0.817"],
          ["Liu 2025", "NHANES 1999-2020, undiagnosed", "AutoML 0.909"],
          ["This work (weighted)", "NHANES 2017-18, undiagnosed", "LR 0.788; ADA 0.734"]], 0.6, 1.4, 12.1, [3.2, 5.0, 3.9], size=16)
bullets(s, ["Our ADA AUC matches De Silva et al. (0.734 vs 0.737); like them, no significant ML gain",
            "None of the benchmarks used survey weights, so their AUCs describe the sample, not the US population"], y=5.0, size=16)

# 14 Conclusion
s = slide("Limitations, conclusion and code", 14)
bullets(s, ["Limitations: cross-sectional; HbA1c-only diabetes; single-visit BP; no public geography; ML bootstrap ignores clustering",
            "Two data bugs found and fixed (prescription placeholder row, COPD variable name)",
            "18.0% of adults with diabetes by HbA1c are undiagnosed, and the gap is not income-graded",
            "A weighted logistic regression on 18 non-lab inputs reaches AUC 0.788 vs 0.734 for the ADA score",
            "Next: add fasting glucose, pool 2017-2023 cycles, validate on 2021-2023",
            f"Code and data pipeline: {REPO}"], size=18)

prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")
