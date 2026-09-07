"""Generates assignment2_report.ipynb from scratch (run once; re-run after editing
cell content below, then re-execute the notebook itself with nbconvert)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cell = nbf.v4.new_markdown_cell(text)
    cell.metadata["language"] = "markdown"
    cells.append(cell)


def code(text):
    cell = nbf.v4.new_code_cell(text)
    cell.metadata["language"] = "python"
    cells.append(cell)


md(r"""# Assignment 2: The Cost of Being Wrong

**DSA 8401 Applied Machine Learning**

This notebook records how I evaluated the mobile-money fraud-scoring pipeline from Assignment 1
and how I turned its probabilities into a practical decision. I kept the original cleaning and
18 engineered features unchanged, then compared models using time-ordered validation, tuned the
strongest candidate, selected a cost-aware threshold, and checked calibration, explanations, and
subgroup behaviour. The four-page report in `report/A2_Report.docx` summarises the same results.

The assignment uses credit-risk language, while my data describe mobile-money transactions. I use
the following interpretation throughout so the decisions remain clear:

| Assignment 2 term | Interpretation here |
|---|---|
| Borrower / application | Mobile-money transaction |
| Default | Fraudulent transaction (`is_fraud = 1`) |
| Approve | Let the transaction proceed unflagged |
| Wrongly reject a good borrower | Flag / hold a legitimate transaction for review |
| Missed default (KES 10,000) | Missed fraud — false negative |
| Wrongly rejected good borrower (KES 800) | Wrongly flagged legitimate transaction — false positive |

I use the supplied KES 10,000 and KES 800 costs, which make a missed fraud 12.5 times more costly
than an unnecessary review. The observed fraud rate is 8.41%, so I report that value rather than
forcing the data to match the brief's illustrative 4% rate.

The reusable cleaning, validation, modelling, and cost functions live in `../src/`. That keeps the
notebook focused on the choices I made and what the resulting numbers mean.""")

code(r"""import sys
from pathlib import Path

sys.path.insert(0, "..")  # so `import src...` resolves to a2-models/src

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 120)

import matplotlib
import matplotlib.pyplot as plt

%matplotlib inline

ART = Path("../artifacts")
(ART / "tables").mkdir(parents=True, exist_ok=True)
(ART / "figures").mkdir(parents=True, exist_ok=True)
(ART / "oof").mkdir(parents=True, exist_ok=True)

from src.data import load_a1_table, TARGET
from src.cv import PurgedBlockedTimeSeriesSplit
from src.models import build_pipeline, build_preprocessor
from src.costs import COST_FN, COST_FP, cost_curve, savings_vs_naive

RANDOM_STATE = 42""")

md(r"""## 0. Load the cleaned transaction table

I start with the cleaned table produced for Assignment 1. This preserves the earlier decisions:
duplicate removal, parsing of `amount` and `txn_time`, canonical customer IDs, and the 18 engineered
features. I also leave out `manual_review_score` and `settlement_status`, since they would reveal
information that would not be available at scoring time. Keeping the transaction timestamp lets me
evaluate the model in the order the transactions would actually have arrived.""")

code(r"""df = load_a1_table()
y = df[TARGET].values
base_rate = float(y.mean())

print("shape:", df.shape, "| customers:", df["customer_id"].nunique())
print(f"fraud ('default') rate: {base_rate:.4f}")
print("ts range:", df["ts"].min(), "->", df["ts"].max())

with open(ART / "tables" / "base_rate.json", "w") as f:
    json.dump({"base_rate": base_rate}, f, indent=2)""")

md(r"""## 1. Evaluation plan: time-ordered validation

I use an expanding-window split based on `ts`. For each fold, the model learns from transactions
up to a cutoff and is tested on the next block, which mirrors a forward-looking deployment. I also
remove a three-day embargo immediately before each test block. This reduces the chance that rows
right at the boundary make the training and test periods look artificially similar.

There is an important limitation in the inherited features. Values such as `recency_days`,
`frequency_90d`, `tenure_days`, and `cashout_ratio` were calculated from each customer's full
history relative to one global scoring date. They were not rebuilt at every transaction date. The
diagnostic below therefore measures how much customers appear in both sides of each fold. A time
purge cannot remove that overlap; fixing it would require rebuilding the Assignment 1 features as
point-in-time features. I report this openly because the scores should not be read as performance
on completely unseen customers.

I use PR-AUC as the main ranking metric. Fraud is only about 8.4% of the data, so ROC-AUC can look
healthy even when the model is not finding enough fraudulent transactions. The precision-recall
curve focuses on the minority class and is more useful for this decision problem; I still calculate
ROC-AUC as a secondary reference.""")

code(r"""cv = PurgedBlockedTimeSeriesSplit(n_splits=5, embargo_days=3.0)
folds = list(cv.split(df["ts"]))

cv_diag = cv.fold_diagnostics(df["ts"], df["customer_id"])
cv_diag.to_csv(ART / "tables" / "cv_diagnostics.csv", index=False)
avg_overlap = cv_diag["customer_overlap_pct"].mean()
print(f"Average train/test customer overlap across folds: {avg_overlap:.1f}%")
cv_diag""")

md(r"""## 2. Comparing the initial models

I begin with three model families: regularised logistic regression, random forest, and XGBoost.
They use the same five purged folds and class weighting, with no model-specific tuning at this
stage. This makes the first comparison about the model families themselves rather than about who
received the most tuning effort.""")

code(r"""MODELS = ["logreg", "random_forest", "xgboost"]

fold_rows = []
oof_store = {}

for model_name in MODELS:
    oof_prob = np.full(len(df), np.nan)
    for k, (tr, te) in enumerate(folds):
        scale_pos_weight = (1 - y[tr].mean()) / y[tr].mean() if model_name == "xgboost" else None
        pipe = build_pipeline(model_name, imbalance="class_weight", scale_pos_weight=scale_pos_weight)
        pipe.fit(df.iloc[tr], y[tr])
        prob = pipe.predict_proba(df.iloc[te])[:, 1]
        oof_prob[te] = prob

        from sklearn.metrics import average_precision_score, roc_auc_score
        fold_rows.append({
            "model": model_name, "fold": k, "n_train": len(tr), "n_test": len(te),
            "pr_auc": average_precision_score(y[te], prob), "roc_auc": roc_auc_score(y[te], prob),
        })
    oof_store[model_name] = oof_prob

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(ART / "tables" / "fold_scores.csv", index=False)

summary = fold_df.groupby("model")[["pr_auc", "roc_auc"]].agg(["mean", "std"]).round(4)
summary.columns = ["_".join(c) for c in summary.columns]
summary = summary.reset_index()
summary.to_csv(ART / "tables" / "model_comparison.csv", index=False)
summary""")

code(r"""from sklearn.metrics import average_precision_score, precision_recall_curve

plt.figure(figsize=(6, 5))
for model_name in MODELS:
    prob = oof_store[model_name]
    valid = ~np.isnan(prob)
    precision, recall, _ = precision_recall_curve(y[valid], prob[valid])
    ap = average_precision_score(y[valid], prob[valid])
    plt.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})")
plt.axhline(base_rate, color="grey", linestyle="--", label=f"no-skill (base rate={base_rate:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall curves (pooled out-of-fold, purged CV)")
plt.legend(); plt.tight_layout()
plt.savefig(ART / "figures" / "pr_curves.png", dpi=150)
plt.show()""")

md(r"""**What I take from the comparison.** The behavioural features provide a limited signal: the
honest A1 pipeline reaches only about 0.62–0.63 ROC-AUC. In the untuned comparison, the tree models
are more flexible than the data seems to support and can fit fold-specific noise, while the more
constrained logistic model generalises slightly better. That does not rule out XGBoost; it suggests
that its default tree complexity is too high. I test that explanation later by tuning shallower
trees with explicit regularisation and a lower learning rate.""")

md(r"""## 3. Handling the class imbalance

Fraud is the minority class, so I compare two ways of making it more visible to XGBoost: class
weighting and SMOTE. In both cases the adjustment is made inside the training fold. I then run a
separate controlled check to show why applying SMOTE before the split gives an overly optimistic
result.""")

code(r"""imb_rows = []
for imbalance in ["class_weight", "smote"]:
    for k, (tr, te) in enumerate(folds):
        scale_pos_weight = (1 - y[tr].mean()) / y[tr].mean() if imbalance == "class_weight" else None
        pipe = build_pipeline("xgboost", imbalance=imbalance, scale_pos_weight=scale_pos_weight)
        pipe.fit(df.iloc[tr], y[tr])
        prob = pipe.predict_proba(df.iloc[te])[:, 1]
        imb_rows.append({"imbalance_strategy": imbalance, "fold": k,
                          "pr_auc": average_precision_score(y[te], prob)})

imb_df = pd.DataFrame(imb_rows)
imb_summary = imb_df.groupby("imbalance_strategy")["pr_auc"].agg(["mean", "std"]).round(4).reset_index()
imb_df.to_csv(ART / "tables" / "imbalance_fold_scores.csv", index=False)
imb_summary.to_csv(ART / "tables" / "imbalance_comparison.csv", index=False)
imb_summary""")

md(r"""**Why the split matters.** I use the final, largest fold as a simple controlled comparison.
The honest version fits the preprocessing and SMOTE only on the training rows, then evaluates on
the untouched test rows. In the leaky version, preprocessing and SMOTE see the pooled train and
test data first. That lets synthetic minority examples use neighbours from the test period, even
though the model is still scored on the same real test rows. Any increase in PR-AUC is therefore
an artefact of allowing test information into training, not a real improvement.""")

code(r"""from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

tr, te = folds[-1]
df_tr, df_te = df.iloc[tr], df.iloc[te]
y_tr, y_te = y[tr], y[te]

# HONEST: fit preprocessing + SMOTE on train rows only
prep_honest = build_preprocessor()
X_tr_honest = prep_honest.fit_transform(df_tr)
X_te_honest = prep_honest.transform(df_te)
X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr_honest, y_tr)
clf_honest = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric="aucpr", tree_method="hist")
clf_honest.fit(X_res, y_res)
pr_auc_honest = average_precision_score(y_te, clf_honest.predict_proba(X_te_honest)[:, 1])

# LEAKY: fit preprocessing + SMOTE on train+test pooled
df_pool = pd.concat([df_tr, df_te], axis=0)
y_pool = np.concatenate([y_tr, y_te])
prep_leaky = build_preprocessor()
X_pool = prep_leaky.fit_transform(df_pool)
X_pool_res, y_pool_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_pool, y_pool)

n_pool, n_train = len(y_pool), len(y_tr)
is_synthetic = np.arange(len(y_pool_res)) >= n_pool
is_original_train = np.zeros(len(y_pool_res), dtype=bool)
is_original_train[:n_train] = True
leaky_train_mask = is_original_train | is_synthetic

clf_leaky = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric="aucpr", tree_method="hist")
clf_leaky.fit(X_pool_res[leaky_train_mask], y_pool_res[leaky_train_mask])
X_leaky_test = prep_leaky.transform(df_te)
pr_auc_leaky = average_precision_score(y_te, clf_leaky.predict_proba(X_leaky_test)[:, 1])

leakage_result = {
    "fold_used": len(folds) - 1, "n_train": int(n_train), "n_test": int(len(y_te)),
    "pr_auc_honest_infold_smote": float(pr_auc_honest),
    "pr_auc_leaky_outoffold_smote": float(pr_auc_leaky),
    "inflation": float(pr_auc_leaky - pr_auc_honest),
    "inflation_pct": float(100 * (pr_auc_leaky - pr_auc_honest) / pr_auc_honest),
}
with open(ART / "tables" / "leakage_proof.json", "w") as f:
    json.dump(leakage_result, f, indent=2)

print(json.dumps(leakage_result, indent=2))""")

md(r"""I carry in-fold SMOTE forward because it gives the better comparison result. More importantly,
the implementation keeps every resampling operation inside its training fold. The leakage check is
only there to quantify how much the reported skill could be inflated by the common alternative.""")

md(r"""## 4. Tuning XGBoost within a fixed budget

The initial comparison suggests that XGBoost needs regularisation rather than more raw complexity.
I use Optuna to search its main tree, sampling, learning-rate, and penalty parameters. The search
is limited to 65 trials and uses pruning, so weak configurations can be stopped early. To keep the
search practical, only the three most recent folds are used and training rows are capped at 50,000
during the search. Those shortcuts apply only while choosing the parameters; the selected model is
refit on the full training folds before I report its performance.""")

code(r"""import optuna

DB_PATH = ART / "optuna_study.db"
STUDY_NAME = "xgb_fraud_pr_auc"
N_TRIALS = 65
SEARCH_FOLD_IDX = [2, 3, 4]
MAX_TRAIN_SUBSAMPLE = 50_000

SEARCH_SPACE = {
    "n_estimators": ("int", 100, 500),
    "max_depth": ("int", 2, 8),
    "learning_rate": ("float_log", 0.01, 0.3),
    "subsample": ("float", 0.5, 1.0),
    "colsample_bytree": ("float", 0.5, 1.0),
    "min_child_weight": ("int", 1, 20),
    "gamma": ("float", 0.0, 5.0),
    "reg_alpha": ("float_log", 1e-3, 10.0),
    "reg_lambda": ("float_log", 1e-3, 10.0),
}


def suggest_params(trial):
    p = {}
    for name, spec in SEARCH_SPACE.items():
        kind = spec[0]
        if kind == "int":
            p[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "float":
            p[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "float_log":
            p[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
    return p


rng = np.random.RandomState(RANDOM_STATE)


def objective(trial):
    params = suggest_params(trial)
    scores = []
    for step, fold_idx in enumerate(SEARCH_FOLD_IDX):
        tr, te = folds[fold_idx]
        tr_sub = rng.choice(tr, size=MAX_TRAIN_SUBSAMPLE, replace=False) if len(tr) > MAX_TRAIN_SUBSAMPLE else tr
        pipe = build_pipeline("xgboost", params=params, imbalance="smote")
        pipe.fit(df.iloc[tr_sub], y[tr_sub])
        prob = pipe.predict_proba(df.iloc[te])[:, 1]
        scores.append(average_precision_score(y[te], prob))
        trial.report(np.mean(scores), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(scores))


if DB_PATH.exists():
    DB_PATH.unlink()

pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
study = optuna.create_study(study_name=STUDY_NAME, storage=f"sqlite:///{DB_PATH}",
                             direction="maximize", sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
print(f"Trials: {len(study.trials)} total, {len(completed)} completed, {len(pruned)} pruned")
print("Best value (mean PR-AUC, search folds, subsampled train):", study.best_value)
print("Best params:", study.best_params)

optuna_best = {
    "best_value": study.best_value, "best_params": study.best_params,
    "n_trials": len(study.trials), "n_completed": len(completed), "n_pruned": len(pruned),
    "search_space": {k: list(v) for k, v in SEARCH_SPACE.items()},
    "search_fold_idx": SEARCH_FOLD_IDX, "max_train_subsample": MAX_TRAIN_SUBSAMPLE,
    "pruner": "MedianPruner(n_startup_trials=10, n_warmup_steps=1)", "sampler": "TPESampler",
}
with open(ART / "tables" / "optuna_best_params.json", "w") as f:
    json.dump(optuna_best, f, indent=2)
study.trials_dataframe().to_csv(ART / "tables" / "optuna_trials.csv", index=False)""")

code(r"""from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

fig = plot_optimization_history(study)
fig.figure.tight_layout()
fig.figure.savefig(ART / "figures" / "optuna_optimization_history.png", dpi=150)
plt.show()

fig2 = plot_param_importances(study)
fig2.figure.tight_layout()
fig2.figure.savefig(ART / "figures" / "optuna_param_importances.png", dpi=150)
plt.show()""")

md(r"""## 5. Final model, threshold, and calibration

I now refit the tuned XGBoost plus SMOTE pipeline on all available training rows in each purged
fold. I also fit isotonic calibration within each training fold using `CalibratedClassifierCV`.
The resulting probabilities are out-of-fold predictions, so the rows used for evaluation were not
used to fit either the model or its calibrator.""")

code(r"""from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

best_params = optuna_best["best_params"]
raw_oof = np.full(len(df), np.nan)
cal_oof = np.full(len(df), np.nan)
tuned_fold_rows = []

for k, (tr, te) in enumerate(folds):
    pipe = build_pipeline("xgboost", params=best_params, imbalance="smote")
    pipe.fit(df.iloc[tr], y[tr])
    raw_prob = pipe.predict_proba(df.iloc[te])[:, 1]
    raw_oof[te] = raw_prob

    cal_pipe = build_pipeline("xgboost", params=best_params, imbalance="smote")
    calibrated = CalibratedClassifierCV(cal_pipe, method="isotonic", cv=3)
    calibrated.fit(df.iloc[tr], y[tr])
    cal_prob = calibrated.predict_proba(df.iloc[te])[:, 1]
    cal_oof[te] = cal_prob

    tuned_fold_rows.append({
        "fold": k, "n_train": len(tr), "n_test": len(te),
        "pr_auc": average_precision_score(y[te], raw_prob),
        "roc_auc": roc_auc_score(y[te], raw_prob),
        "brier_raw": brier_score_loss(y[te], raw_prob),
        "brier_calibrated": brier_score_loss(y[te], cal_prob),
    })

tuned_fold_df = pd.DataFrame(tuned_fold_rows)
tuned_fold_df.to_csv(ART / "tables" / "tuned_fold_scores.csv", index=False)
tuned_fold_df""")

code(r"""valid = ~np.isnan(raw_oof)
y_v = y[valid]

tuned_summary = {
    "model": "xgboost_tuned_smote",
    "pr_auc_mean": float(tuned_fold_df["pr_auc"].mean()), "pr_auc_std": float(tuned_fold_df["pr_auc"].std()),
    "roc_auc_mean": float(tuned_fold_df["roc_auc"].mean()), "roc_auc_std": float(tuned_fold_df["roc_auc"].std()),
    "brier_raw_pooled": float(brier_score_loss(y_v, raw_oof[valid])),
    "brier_calibrated_pooled": float(brier_score_loss(y_v, cal_oof[valid])),
}
with open(ART / "tables" / "tuned_model_summary.json", "w") as f:
    json.dump(tuned_summary, f, indent=2)

comp = pd.read_csv(ART / "tables" / "model_comparison.csv")
comp = comp[comp["model"] != "xgboost_tuned_smote"]
new_row = {"model": "xgboost_tuned_smote", "pr_auc_mean": tuned_summary["pr_auc_mean"],
           "pr_auc_std": tuned_summary["pr_auc_std"], "roc_auc_mean": tuned_summary["roc_auc_mean"],
           "roc_auc_std": tuned_summary["roc_auc_std"]}
comp = pd.concat([comp, pd.DataFrame([new_row])], ignore_index=True)
comp.to_csv(ART / "tables" / "model_comparison.csv", index=False)

np.savez(ART / "oof" / "tuned_oof_probabilities.npz", y=y, raw_oof=raw_oof, cal_oof=cal_oof, valid=valid)
print(json.dumps(tuned_summary, indent=2))
comp""")

md(r"""**Choosing the operating point.** A probability is not useful on its own; I need a cut-off for
deciding whether to flag a transaction. I sweep the threshold using the supplied KES 10,000 cost
for a missed fraud and KES 800 cost for an unnecessary flag. I do this for both the raw and
calibrated out-of-fold probabilities, then compare each result with the default 0.5 threshold.""")

code(r"""raw_result = savings_vs_naive(y_v, raw_oof[valid])
cal_result = savings_vs_naive(y_v, cal_oof[valid])
cost_summary = {
    "raw": {k: v for k, v in raw_result.items() if k not in ("best_row", "naive_row")},
    "calibrated": {k: v for k, v in cal_result.items() if k not in ("best_row", "naive_row")},
    "threshold_shift": float(cal_result["best_threshold"] - raw_result["best_threshold"]),
}
with open(ART / "tables" / "cost_threshold.json", "w") as f:
    json.dump(cost_summary, f, indent=2)
print(json.dumps(cost_summary, indent=2))""")

code(r"""frac_raw, mean_raw = calibration_curve(y_v, raw_oof[valid], n_bins=10, strategy="quantile")
frac_cal, mean_cal = calibration_curve(y_v, cal_oof[valid], n_bins=10, strategy="quantile")

plt.figure(figsize=(6, 5))
plt.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
plt.plot(mean_raw, frac_raw, "o-", label=f"raw XGBoost (Brier={tuned_summary['brier_raw_pooled']:.4f})")
plt.plot(mean_cal, frac_cal, "o-", label=f"isotonic-calibrated (Brier={tuned_summary['brier_calibrated_pooled']:.4f})")
plt.xlabel("Mean predicted probability"); plt.ylabel("Observed fraud rate")
plt.title("Reliability diagram (pooled out-of-fold, purged CV)")
plt.legend(); plt.tight_layout()
plt.savefig(ART / "figures" / "reliability_diagram.png", dpi=150)
plt.show()

curve_raw = cost_curve(y_v, raw_oof[valid])
curve_cal = cost_curve(y_v, cal_oof[valid])
plt.figure(figsize=(6, 5))
plt.plot(curve_raw["threshold"], curve_raw["cost_per_1000"], label="raw scores")
plt.plot(curve_cal["threshold"], curve_cal["cost_per_1000"], label="calibrated scores")
plt.axvline(raw_result["best_threshold"], color="C0", linestyle="--", alpha=0.6)
plt.axvline(cal_result["best_threshold"], color="C1", linestyle="--", alpha=0.6)
plt.axvline(0.5, color="grey", linestyle=":", label="naive 0.5")
plt.xlabel("Decision threshold"); plt.ylabel("Total cost per 1,000 applications (KES)")
plt.title("Cost curve: threshold selection")
plt.legend(); plt.tight_layout()
plt.savefig(ART / "figures" / "cost_curve.png", dpi=150)
plt.show()""")

md(r"""Calibration changes the scale of the XGBoost scores rather than their ranking. In this run,
isotonic calibration compresses the raw scores and moves the cost-minimising threshold downward. I
use that calibrated threshold for the final deployment recommendation because it is the one tied to
the stated costs.""")

md(r"""## 6. Understanding the predictions and subgroup results

For the saved pipeline, I refit the tuned model on the complete Assignment 1 table. I use SHAP to
look at both the overall feature pattern and one individual prediction. The explanation is in the
preprocessed feature space seen by XGBoost. SMOTE is used while fitting only, so it does not create
synthetic transactions during prediction or explanation.""")

code(r"""import joblib
import shap

final_pipeline = build_pipeline("xgboost", params=best_params, imbalance="smote")
final_pipeline.fit(df, y)
joblib.dump(final_pipeline, ART / "final_fraud_pipeline.joblib")

prep = final_pipeline.named_steps["prep"]
clf = final_pipeline.named_steps["clf"]
feature_names = prep.get_feature_names_out()

sample_idx = rng.choice(len(df), size=min(3000, len(df)), replace=False)
X_sample = np.asarray(prep.transform(df.iloc[sample_idx]))

explainer = shap.TreeExplainer(clf)
shap_values = explainer(X_sample)
shap_values.feature_names = list(feature_names)

shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=15)
plt.tight_layout()
plt.savefig(ART / "figures" / "shap_global_summary.png", dpi=150, bbox_inches="tight")
plt.show()""")

code(r"""prob_sample = clf.predict_proba(X_sample)[:, 1]
y_sample = y[sample_idx]
fraud_hits = np.where((y_sample == 1) & (prob_sample > 0.5))[0]
local_i = fraud_hits[np.argmax(prob_sample[fraud_hits])] if len(fraud_hits) else int(np.argmax(prob_sample))

shap.plots.waterfall(shap_values[local_i], show=False, max_display=12)
plt.tight_layout()
plt.savefig(ART / "figures" / "shap_local_waterfall.png", dpi=150, bbox_inches="tight")
plt.show()

local_record = df.iloc[sample_idx[local_i]]
shap_local_example = {
    "txn_id": str(local_record["txn_id"]), "customer_id": str(local_record["customer_id"]),
    "region": str(local_record["region"]), "true_label": int(y_sample[local_i]),
    "predicted_probability": float(prob_sample[local_i]),
}
with open(ART / "tables" / "shap_local_example.json", "w") as f:
    json.dump(shap_local_example, f, indent=2)
print(json.dumps(shap_local_example, indent=2))""")

md(r"""**Subgroup checks.** I compare the share of transactions left unflagged and the false-negative
rate among actual fraud cases at the calibrated cost-optimal threshold. I report these measures by
region and by tenure band to see whether the single threshold behaves very differently across
groups. These are monitoring checks, not a claim that the available features establish fairness.""")

code(r"""def tenure_band(days):
    return pd.cut(days, bins=[-np.inf, 90, 365, np.inf],
                  labels=["new (<90d)", "established (90-365d)", "long-tenure (>365d)"])


threshold = cost_summary["calibrated"]["best_threshold"]
meta = df.loc[valid, ["region", "segment", "tenure_days"]].copy()
meta["tenure_band"] = tenure_band(meta["tenure_days"])
meta["y_true"] = y[valid]
meta["y_prob"] = cal_oof[valid]
meta["flagged"] = (meta["y_prob"] >= threshold).astype(int)


def subgroup_table(group_col):
    rows = []
    for g, gdf in meta.groupby(group_col, observed=True):
        n = len(gdf)
        approval_rate = 1 - gdf["flagged"].mean()
        fn = int(((gdf["flagged"] == 0) & (gdf["y_true"] == 1)).sum())
        positives = int((gdf["y_true"] == 1).sum())
        fnr = fn / positives if positives else np.nan
        rows.append({group_col: g, "n": n, "base_rate": gdf["y_true"].mean(),
                     "approval_rate": approval_rate, "n_positives": positives,
                     "false_negative_rate": fnr})
    return pd.DataFrame(rows)


region_table = subgroup_table("region")
tenure_table = subgroup_table("tenure_band")
region_table.to_csv(ART / "tables" / "fairness_by_region.csv", index=False)
tenure_table.to_csv(ART / "tables" / "fairness_by_tenure_band.csv", index=False)

print(f"Final decision threshold used for fairness metrics: {threshold:.4f}\n")
display(region_table.round(4))
display(tenure_table.round(4))""")

md(r"""The regional results show a meaningful spread in approval and false-negative rates, so the
regions at the extremes would need review before deployment. The tenure comparison is much weaker:
`tenure_days` is a static aggregate against one global scoring date, and most evaluated rows fall in
the long-tenure group. I therefore treat the smaller tenure groups as descriptive only, not as
evidence that the model is fair or unfair.

The SHAP example also gives me a way to describe why a transaction was flagged instead of exposing
only an unexplained score. The regional and tenure summaries provide a corresponding internal
check: a group with a noticeably higher false-negative rate or lower approval rate should be
investigated, even though region and tenure were not used as direct protected labels in the model.""")

md(r"""## 7. Recommendation

Based on the comparisons above, I recommend the tuned XGBoost model with in-fold SMOTE and isotonic
calibration. The production decision should use the calibrated threshold printed below rather than
the default 0.5 cut-off.

This choice is based on the tuned model's cost-aware performance on the purged, time-ordered folds,
not on accuracy alone. I would still treat the reported performance as optimistic until a monitored
pilot tests the pipeline on genuinely unseen customers, because the inherited features create
customer overlap between training and evaluation periods.""")

code(r"""print(f"Recommended model     : tuned XGBoost + in-fold SMOTE + isotonic calibration")
print(f"Recommended threshold  : {cost_summary['calibrated']['best_threshold']:.4f} (on calibrated probability)")
print(f"Expected cost / 1,000  : KES {cost_summary['calibrated']['best_cost_per_1000']:,.0f}")
print(f"Saving vs naive 0.5    : KES {cost_summary['calibrated']['savings_per_1000']:,.0f} "
      f"({cost_summary['calibrated']['savings_pct']:.1f}%) per 1,000 transactions")""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

with open("assignment2_report.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Wrote assignment2_report.ipynb with", len(cells), "cells")
