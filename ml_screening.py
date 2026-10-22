"""
Undiagnosed-diabetes screening models on NHANES 2017-2018
=========================================================
Task: among adults (20+) who have NOT been told they have diabetes, predict
laboratory-defined diabetes (HbA1c >= 6.5%) from non-laboratory features
that could be collected at a screening desk.

Models compared (nested, stratified cross-validation, NHANES MEC weights used
as sample weights in both fitting and evaluation):
  - Adapted ADA diabetes risk score (rule-based baseline)
  - Logistic regression (L2)
  - Random forest
  - Histogram gradient boosting (LightGBM-style)
  - Multilayer perceptron

Outputs (out/):
  ml_cv_metrics.csv       per-fold metrics for every model
  ml_summary.csv          mean / SD across folds + bootstrap CI on pooled OOF AUC
  ml_best_params.csv      inner-CV hyperparameters chosen in each outer fold
  ml_oof_predictions.csv  out-of-fold scores for every respondent
  ml_permutation_importance.csv
Figures (figures/): ml_roc.png, ml_pr.png, ml_calibration.png, ml_importance.png
"""
import importlib.util
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
OUTER_K, INNER_K = 5, 3

spec = importlib.util.spec_from_file_location("pipe", "nhanes_kpi_pipeline.py")
pipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipe)


# ----------------------------------------------------------------------------
# 1. Cohort and features
# ----------------------------------------------------------------------------
def build_cohort() -> pd.DataFrame:
    d = pipe.derive(pipe.build_analytic_file())
    # Not previously diagnosed: DIQ010 == 2 (No). Borderline (3) is treated as
    # diagnosed, matching the primary DM_UNDIAG definition, so it is excluded.
    c = d[(d["AGE"] >= 20) & (d["DIQ010"] == 2) & d["LBXGH"].notna()
          & (d["WTMEC2YR"] > 0)].copy()
    c["y"] = (c["LBXGH"] >= 6.5).astype(int)

    yn = lambda s: s.where(s.isin([1, 2])).map({1.0: 1.0, 2.0: 0.0})
    c["MALE"] = (c["RIAGENDR"] == 1).astype(float)
    c["FAM_HIST_DM"] = yn(c["MCQ300C"])
    c["TOLD_HBP"] = yn(c["BPQ020"])
    c["TOLD_HICHOL"] = yn(c["BPQ080"])
    c["WHTR"] = c["BMXWAIST"] / c["BMXHT"]
    c["SRH"] = c["HUQ010"].where(c["HUQ010"] <= 5)
    c["EDUC"] = c["DMDEDUC2"].where(c["DMDEDUC2"] <= 5)
    c["INSURED_F"] = c["INSURED"]
    c["USUAL_CARE_F"] = c["HAS_USUAL_CARE"]
    return c


NUM = ["AGE", "INDFMPIR", "EDUC", "BMXBMI", "BMXWAIST", "WHTR", "SBP", "DBP",
       "SRH", "PHQ9", "RX_N"]
BIN = ["MALE", "FAM_HIST_DM", "TOLD_HBP", "TOLD_HICHOL", "INSURED_F", "USUAL_CARE_F"]
CAT = ["RACE_ETH"]
FEATURES = NUM + BIN + CAT


def ada_score(c: pd.DataFrame) -> np.ndarray:
    """ADA diabetes risk test (Bang et al., 2009), adapted: 5 of 7 items.
    Gestational diabetes and physical activity are not in the downloaded
    components, so they score 0 for everyone (max score 9 instead of 11)."""
    age = c["AGE"]
    s = np.select([age < 40, age < 50, age < 60], [0, 1, 2], 3)
    s = s + c["MALE"].fillna(0)
    s = s + c["FAM_HIST_DM"].fillna(0)
    s = s + c["TOLD_HBP"].fillna(0)
    bmi = c["BMXBMI"]
    s = s + np.where(bmi.isna(), 0, np.select([bmi < 25, bmi < 30, bmi < 40], [0, 1, 2], 3))
    return np.asarray(s, dtype=float)


def preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM),
        ("bin", SimpleImputer(strategy="most_frequent"), BIN),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])


MODELS = {
    "Logistic regression": (
        lambda: Pipeline([("pre", preprocessor(True)),
                          ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))]),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]}),
    "Random forest": (
        lambda: Pipeline([("pre", preprocessor(False)),
                          ("clf", RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                                         class_weight="balanced_subsample",
                                                         random_state=SEED))]),
        {"clf__max_depth": [4, 8, None], "clf__min_samples_leaf": [5, 20]}),
    "Gradient boosting": (
        lambda: Pipeline([("pre", preprocessor(False)),
                          ("clf", HistGradientBoostingClassifier(max_iter=300, early_stopping=False,
                                                                 class_weight="balanced",
                                                                 random_state=SEED))]),
        {"clf__learning_rate": [0.03, 0.1], "clf__max_depth": [2, 3],
         "clf__l2_regularization": [0.0, 1.0]}),
    "MLP": (
        lambda: Pipeline([("pre", preprocessor(True)),
                          ("clf", MLPClassifier(max_iter=600, early_stopping=True,
                                                random_state=SEED))]),
        {"clf__hidden_layer_sizes": [(16,), (32, 16)], "clf__alpha": [1e-3, 1e-1]}),
}


def fit(model, X, y, w):
    name = type(model.named_steps["clf"]).__name__
    if name == "MLPClassifier":          # MLP has no sample_weight support
        return model.fit(X, y)
    return model.fit(X, y, clf__sample_weight=w)


def wauc(y, s, w):
    return roc_auc_score(y, s, sample_weight=w)


def sens_at_spec(y, s, w, spec=0.80):
    fpr, tpr, _ = roc_curve(y, s, sample_weight=w)
    return float(np.interp(1 - spec, fpr, tpr))


def metrics(y, s, w, prob=None):
    out = dict(auc_w=wauc(y, s, w), auc_unw=roc_auc_score(y, s),
               auprc_w=average_precision_score(y, s, sample_weight=w),
               sens_at_spec80_w=sens_at_spec(y, s, w))
    out["brier_w"] = brier_score_loss(y, prob, sample_weight=w) if prob is not None else np.nan
    return out


# ----------------------------------------------------------------------------
# 2. Nested CV
# ----------------------------------------------------------------------------
def main():
    c = build_cohort()
    X, y, w = c[FEATURES], c["y"].to_numpy(), c["WTMEC2YR"].to_numpy()
    w = w / w.mean()
    print(f"Cohort: n={len(c):,}, positives={y.sum()} "
          f"(unweighted {y.mean()*100:.2f}%, weighted {np.average(y, weights=w)*100:.2f}%)")

    outer = StratifiedKFold(OUTER_K, shuffle=True, random_state=SEED)
    oof = pd.DataFrame({"SEQN": c["SEQN"].to_numpy(), "y": y, "w": w,
                        "ADA score": ada_score(c)})
    rows, best_rows = [], []

    for fold, (tr, te) in enumerate(outer.split(X, y), 1):
        # ADA baseline: fixed rule, evaluated on the same test fold
        s = oof.loc[te, "ADA score"].to_numpy()
        m = metrics(y[te], s, w[te])
        hi = s >= 5
        m["sens_cut5"] = np.average(hi[y[te] == 1], weights=w[te][y[te] == 1])
        m["spec_cut5"] = np.average(~hi[y[te] == 0], weights=w[te][y[te] == 0])
        rows.append(dict(model="ADA score (adapted)", fold=fold, **m))

        inner = StratifiedKFold(INNER_K, shuffle=True, random_state=SEED + fold)
        for name, (make, grid) in MODELS.items():
            keys, best, best_auc = list(grid), None, -1
            for vals in itertools.product(*grid.values()):
                params = dict(zip(keys, vals))
                aucs = []
                for itr, ite in inner.split(X.iloc[tr], y[tr]):
                    a, b = tr[itr], tr[ite]
                    mdl = fit(make().set_params(**params), X.iloc[a], y[a], w[a])
                    aucs.append(wauc(y[b], mdl.predict_proba(X.iloc[b])[:, 1], w[b]))
                if np.mean(aucs) > best_auc:
                    best_auc, best = np.mean(aucs), params
            mdl = fit(make().set_params(**best), X.iloc[tr], y[tr], w[tr])
            p = mdl.predict_proba(X.iloc[te])[:, 1]
            oof.loc[te, name] = p
            rows.append(dict(model=name, fold=fold, **metrics(y[te], p, w[te], p)))
            best_rows.append(dict(model=name, fold=fold, inner_auc_w=round(best_auc, 4),
                                  **{k.replace("clf__", ""): v for k, v in best.items()}))
            print(f"  fold {fold} {name:20s} AUC_w={rows[-1]['auc_w']:.3f}  best={best}")

    cv = pd.DataFrame(rows)
    cv.to_csv(pipe.OUT / "ml_cv_metrics.csv", index=False)
    pd.DataFrame(best_rows).to_csv(pipe.OUT / "ml_best_params.csv", index=False)
    oof.to_csv(pipe.OUT / "ml_oof_predictions.csv", index=False)

    # Bootstrap CI on pooled out-of-fold weighted AUC, and AUC difference vs ADA
    rng = np.random.default_rng(SEED)
    names = ["ADA score (adapted)"] + list(MODELS)
    col = {"ADA score (adapted)": "ADA score", **{n: n for n in MODELS}}
    boots = {n: [] for n in names}
    diffs = {n: [] for n in MODELS}
    idx = np.arange(len(oof))
    for _ in range(1000):
        bi = rng.choice(idx, len(idx), replace=True)
        yb, wb = oof["y"].to_numpy()[bi], oof["w"].to_numpy()[bi]
        if yb.sum() == 0:
            continue
        a_ada = wauc(yb, oof[col["ADA score (adapted)"]].to_numpy()[bi], wb)
        boots["ADA score (adapted)"].append(a_ada)
        for n in MODELS:
            a = wauc(yb, oof[n].to_numpy()[bi], wb)
            boots[n].append(a)
            diffs[n].append(a - a_ada)

    summ = []
    for n in names:
        sub = cv[cv.model == n]
        pooled = wauc(oof["y"], oof[col[n]], oof["w"])
        r = dict(model=n,
                 auc_w_mean=sub.auc_w.mean(), auc_w_sd=sub.auc_w.std(),
                 auc_unw_mean=sub.auc_unw.mean(),
                 auprc_w_mean=sub.auprc_w.mean(), auprc_w_sd=sub.auprc_w.std(),
                 sens_at_spec80_w_mean=sub.sens_at_spec80_w.mean(),
                 brier_w_mean=sub.brier_w.mean(),
                 pooled_oof_auc_w=pooled,
                 pooled_auc_ci_lo=np.percentile(boots[n], 2.5),
                 pooled_auc_ci_hi=np.percentile(boots[n], 97.5))
        if n in diffs:
            dd = np.array(diffs[n])
            r.update(delta_auc_vs_ada=np.mean(dd), delta_ci_lo=np.percentile(dd, 2.5),
                     delta_ci_hi=np.percentile(dd, 97.5))
        else:
            r.update(sens_cut5_mean=sub.sens_cut5.mean(), spec_cut5_mean=sub.spec_cut5.mean())
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.round(4).to_csv(pipe.OUT / "ml_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))

    importance_and_figures(c, X, y, w, oof, summ, pd.DataFrame(best_rows), col, names)


def importance_and_figures(c, X, y, w, oof, summ, best_rows, col, names):
    """Permutation importance for the best model (by pooled OOF weighted AUC),
    refit on all data with its most frequently chosen hyperparameters.
    Descriptive only: it explains the final model, not the CV estimate."""
    best_name = summ[summ.model != "ADA score (adapted)"].sort_values(
        "pooled_oof_auc_w", ascending=False).model.iloc[0]
    bp = best_rows[best_rows.model == best_name].drop(
        columns=["model", "fold", "inner_auc_w"]).dropna(axis=1, how="all")
    params = {}
    for k, v in bp.mode().iloc[0].to_dict().items():
        if k in ("max_depth", "min_samples_leaf") and pd.notna(v):
            v = int(v)
        params[f"clf__{k}"] = None if (not isinstance(v, str) and pd.isna(v)) else v
    make, _ = MODELS[best_name]
    final = fit(make().set_params(**params), X, y, w)
    pi = permutation_importance(final, X, y, scoring="roc_auc", n_repeats=10,
                                random_state=SEED, sample_weight=w)
    imp = (pd.DataFrame({"feature": FEATURES, "importance_mean": pi.importances_mean,
                         "importance_sd": pi.importances_std})
           .sort_values("importance_mean", ascending=False))
    imp["model"] = best_name
    imp.round(5).to_csv(pipe.OUT / "ml_permutation_importance.csv", index=False)
    print(f"Permutation importance ({best_name}, params={params}):")
    print(imp.head(10).round(4).to_string(index=False))
    if best_name == "Logistic regression":
        clf = final.named_steps["clf"]
        names_out = final.named_steps["pre"].get_feature_names_out()
        coef = pd.DataFrame({"term": names_out, "coef_std": clf.coef_[0],
                             "odds_ratio_per_sd": np.exp(clf.coef_[0])})
        coef.round(4).to_csv(pipe.OUT / "ml_lr_coefficients.csv", index=False)

    make_figures(oof, col, names, imp, best_name)


def make_figures(oof, col, names, imp, best_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from pathlib import Path
    Path("figures").mkdir(exist_ok=True)
    colors = dict(zip(names, ["#8C8C8C", "#1f77b4", "#2ca02c", "#d62728", "#9467bd"]))
    y, w = oof["y"], oof["w"]

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for n in names:
        fpr, tpr, _ = roc_curve(y, oof[col[n]], sample_weight=w)
        ax.plot(fpr, tpr, color=colors[n], lw=1.6,
                label=f"{n} (AUC {wauc(y, oof[col[n]], w):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)
    fig.savefig("figures/ml_roc.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for n in names:
        pr, rc, _ = precision_recall_curve(y, oof[col[n]], sample_weight=w)
        ax.plot(rc, pr, color=colors[n], lw=1.6,
                label=f"{n} (AP {average_precision_score(y, oof[col[n]], sample_weight=w):.3f})")
    ax.axhline(np.average(y, weights=w), color="k", ls=":", lw=0.8)
    ax.set_xlabel("Recall (sensitivity)"); ax.set_ylabel("Precision (PPV)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.savefig("figures/ml_pr.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for n in names[1:]:
        ft, mp = calibration_curve(y, oof[n], n_bins=8, strategy="quantile")
        ax.plot(mp, ft, "o-", color=colors[n], lw=1.2, ms=3, label=n)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.savefig("figures/ml_calibration.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    labels = {"RACE_ETH": "Race/ethnicity", "AGE": "Age", "WHTR": "Waist-to-height ratio",
              "SBP": "Systolic BP", "DBP": "Diastolic BP", "FAM_HIST_DM": "Family history of DM",
              "BMXBMI": "BMI", "BMXWAIST": "Waist circumference", "INSURED_F": "Insured",
              "SRH": "Self-rated health", "RX_N": "No. of prescriptions", "INDFMPIR": "Income-to-poverty",
              "EDUC": "Education", "PHQ9": "PHQ-9 score", "MALE": "Male sex",
              "TOLD_HBP": "Told high BP", "TOLD_HICHOL": "Told high cholesterol",
              "USUAL_CARE_F": "Usual source of care"}
    top = imp.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.barh([labels.get(f, f) for f in top.feature], top.importance_mean,
            xerr=top.importance_sd, color="#d62728", alpha=0.8)
    ax.set_xlabel("Drop in weighted AUC when permuted")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig("figures/ml_importance.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def post():
    """Re-run importance + figures from saved CV outputs (no retraining of CV)."""
    c = build_cohort()
    X, y, w = c[FEATURES], c["y"].to_numpy(), c["WTMEC2YR"].to_numpy()
    w = w / w.mean()
    oof = pd.read_csv(pipe.OUT / "ml_oof_predictions.csv")
    summ = pd.read_csv(pipe.OUT / "ml_summary.csv")
    best_rows = pd.read_csv(pipe.OUT / "ml_best_params.csv")
    best_rows["hidden_layer_sizes"] = best_rows["hidden_layer_sizes"].map(
        lambda v: eval(v) if isinstance(v, str) else v)
    names = ["ADA score (adapted)"] + list(MODELS)
    col = {"ADA score (adapted)": "ADA score", **{n: n for n in MODELS}}
    importance_and_figures(c, X, y, w, oof, summ, best_rows, col, names)


if __name__ == "__main__":
    import sys
    post() if "--post" in sys.argv else main()
