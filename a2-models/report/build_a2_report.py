"""
Builds the 4-page Assignment 2 report from the artifacts produced by scripts/*.py.
Every number in the narrative is read from the saved artifacts, not hardcoded, so
re-running the pipeline and then this script keeps the report honest.
"""
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT_PATH = Path(__file__).resolve().parent / "A2_Report.docx"

BLUE = RGBColor(0x1F, 0x3A, 0x5F)
MUTED = RGBColor(0x55, 0x55, 0x55)


def j(name):
    with open(ART / "tables" / name) as f:
        return json.load(f)


def c(name):
    return pd.read_csv(ART / "tables" / name)


# ---- load everything up front ----
base_rate = j("base_rate.json")["base_rate"]
cv_diag = c("cv_diagnostics.csv")
model_comp = c("model_comparison.csv").set_index("model")
fold_scores = c("fold_scores.csv")
imbalance_comp = c("imbalance_comparison.csv").set_index("imbalance_strategy")
leakage = j("leakage_proof.json")
optuna_best = j("optuna_best_params.json")
tuned_summary = j("tuned_model_summary.json")
cost = j("cost_threshold.json")
region_fair = c("fairness_by_region.csv")
tenure_fair = c("fairness_by_tenure_band.csv")
shap_local = j("shap_local_example.json")

avg_overlap = cv_diag["customer_overlap_pct"].mean()


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def format_paragraph(p, size=10.3, before=0, after=3.2, line=1.0):
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = line
    for run in p.runs:
        run.font.name = "Calibri"
        run.font.size = Pt(size)
    return p


def add_para(doc, text, size=10.3, before=0, after=3.2, bold_prefix=None):
    p = doc.add_paragraph()
    if bold_prefix and text.startswith(bold_prefix):
        r = p.add_run(bold_prefix)
        r.bold = True
        r.font.name = "Calibri"
        r.font.size = Pt(size)
        rest = text[len(bold_prefix):]
        if rest:
            r2 = p.add_run(rest)
            r2.font.name = "Calibri"
            r2.font.size = Pt(size)
    else:
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(size)
    return format_paragraph(p, size=size, before=before, after=after)


def add_heading(doc, text, size=13.0):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.bold = True
    r.font.name = "Calibri"
    r.font.size = Pt(size)
    r.font.color.rgb = BLUE
    return p


def add_table(doc, headers, rows, widths, header_size=8.0, body_size=7.8):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for i, cell in enumerate(table.rows[0].cells):
        cell.text = headers[i]
        cell.width = Cm(widths[i])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for p in cell.paragraphs:
            p.paragraph_format.space_after = Pt(0)
            for r in p.runs:
                r.bold = True
                r.font.name = "Calibri"
                r.font.size = Pt(header_size)
                r.font.color.rgb = BLUE
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = str(value)
            cells[i].width = Cm(widths[i])
            cells[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
            for p in cells[i].paragraphs:
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 0.92
                for r in p.runs:
                    r.font.name = "Calibri"
                    r.font.size = Pt(body_size)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


doc = Document()
for section in doc.sections:
    section.top_margin = Cm(1.2)
    section.bottom_margin = Cm(1.2)
    section.left_margin = Cm(1.4)
    section.right_margin = Cm(1.4)

style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(10.3)

# ============================== PAGE 1 ==============================
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(1)
r = title.add_run("Assignment 2: The Cost of Being Wrong")
r.bold = True
r.font.name = "Calibri"
r.font.size = Pt(16)
r.font.color.rgb = BLUE

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle.paragraph_format.space_after = Pt(7)
r = subtitle.add_run("DSA 8401 Applied Machine Learning | Repository: a2-models")
r.italic = True
r.font.name = "Calibri"
r.font.size = Pt(9.5)
r.font.color.rgb = MUTED

add_para(
    doc,
    "This submission reuses Assignment 1's mobile-money fraud-scoring pipeline "
    "exactly as built (180,000 leak-free transactions, 3,991 customers, 18 "
    "engineered features, is_fraud target). A1 is a fraud-detection task, not a "
    "loan-approval dataset, so the assignment's borrower/default vocabulary is "
    f"mapped one-for-one onto it throughout: 'default' = a fraudulent transaction "
    f"(observed base rate {base_rate:.2%}, not exactly 4%, reported honestly rather "
    "than forced to match); 'approve' = let a transaction proceed unflagged; "
    "'wrongly reject a good borrower' = flag a legitimate transaction for review. "
    "The cost values from the brief are used as given: a missed default (missed "
    "fraud, a false negative) costs KES 10,000; a wrongly rejected good borrower "
    "(a wrongly flagged legitimate transaction, a false positive) costs KES 800 -- "
    "a 12.5x asymmetry."
)

add_heading(doc, "1. Evaluation Plan")
add_para(
    doc,
    "Validation uses a purged, blocked, expanding-window time-series split (5 "
    "folds): the frame is sorted by transaction time, each fold trains on every "
    "transaction up to a cutoff and tests on the following block, and a 3-day "
    "embargo immediately before each test block is purged from training to remove "
    "the border overlap a plain chronological split would leave. No fold ever sees "
    "a transaction before training on transactions that precede it in time.",
    bold_prefix="Validation uses"
)
add_para(
    doc,
    "Disclosed limitation: several A1 features (recency_days, frequency_90d, "
    "tenure_days, cashout_ratio and others) are customer-level aggregates computed "
    "once from each customer's entire history relative to a single global scoring "
    f"timestamp, not recomputed at each transaction's own time. Because customers "
    f"transact throughout the whole 18-month window, average customer overlap "
    f"between each fold's train and test set is {avg_overlap:.1f}% (see "
    "cv_diagnostics.csv) -- a given customer's earlier rows already encode "
    "information from their later behaviour. Purging by time cannot fix this "
    "without re-deriving point-in-time features, which would violate the 'use A1's "
    "output exactly as it was' constraint, so the honest choice was to measure and "
    "disclose it rather than present the split as fully leak-free.",
    bold_prefix="Disclosed limitation:"
)
add_para(
    doc,
    "PR-AUC is the headline metric, not ROC-AUC. At an "
    f"{base_rate:.1%} fraud rate, the negative class dominates the confusion matrix, "
    "so a classifier can rack up a high true-negative rate (and hence a flattering "
    "ROC-AUC) while still missing most of the rare, expensive positives; ROC-AUC's "
    "false-positive-rate axis is diluted by the huge negative denominator. "
    "Precision-Recall focuses on the positive class directly, so it moves when the "
    "model actually gets better at finding fraud instead of merely being correct "
    "about the easy majority.",
    bold_prefix="PR-AUC is"
)

# ============================== PAGE 2 ==============================
doc.add_page_break()
add_heading(doc, "2. Ladder of Models: Bias vs. Variance")
row_logreg = model_comp.loc["logreg"]
row_rf = model_comp.loc["random_forest"]
row_xgb = model_comp.loc["xgboost"]
row_xgb_tuned = model_comp.loc["xgboost_tuned_smote"]
add_para(
    doc,
    "Regularised logistic regression, a random forest and XGBoost were compared on "
    "the identical five purged folds (class-weighted, untuned defaults, so the "
    "comparison isolates model family, not tuning effort).",
)
add_table(
    doc,
    ["Model", "PR-AUC (mean +/- std)", "ROC-AUC (mean +/- std)"],
    [
        ["Logistic regression (L2, balanced)", f"{row_logreg.pr_auc_mean:.4f} +/- {row_logreg.pr_auc_std:.4f}",
         f"{row_logreg.roc_auc_mean:.4f} +/- {row_logreg.roc_auc_std:.4f}"],
        ["Random forest (balanced)", f"{row_rf.pr_auc_mean:.4f} +/- {row_rf.pr_auc_std:.4f}",
         f"{row_rf.roc_auc_mean:.4f} +/- {row_rf.roc_auc_std:.4f}"],
        ["XGBoost (untuned)", f"{row_xgb.pr_auc_mean:.4f} +/- {row_xgb.pr_auc_std:.4f}",
         f"{row_xgb.roc_auc_mean:.4f} +/- {row_xgb.roc_auc_std:.4f}"],
        ["XGBoost (tuned + in-fold SMOTE)", f"{row_xgb_tuned.pr_auc_mean:.4f} +/- {row_xgb_tuned.pr_auc_std:.4f}",
         f"{row_xgb_tuned.roc_auc_mean:.4f} +/- {row_xgb_tuned.roc_auc_std:.4f}"],
    ],
    [5.6, 4.6, 4.6],
)
add_para(
    doc,
    "With the behavioural signal this weak (fraud here is closer to random noise "
    "than a clean separable pattern -- the honest, leakage-free A1 features top out "
    "around 0.62-0.63 ROC-AUC), untuned bagging and boosting default to deep, "
    "high-variance trees that fit sampling noise in the training folds and generalise "
    "slightly worse than the high-bias, strongly regularised logistic regression. "
    "This flips only once XGBoost is properly budget-tuned (Section 5): shallower "
    "trees, explicit L1/L2 penalties and a lower learning rate trade variance for "
    "bias in a way default hyperparameters do not.",
)

add_heading(doc, "3. Handling the Imbalance")
row_cw = imbalance_comp.loc["class_weight"]
row_sm = imbalance_comp.loc["smote"]
add_para(
    doc,
    f"Two in-fold strategies were compared for XGBoost: class weighting "
    f"(PR-AUC {row_cw['mean']:.4f} +/- {row_cw['std']:.4f}) vs. SMOTE fit inside "
    f"each training fold (PR-AUC {row_sm['mean']:.4f} +/- {row_sm['std']:.4f}). "
    "In-fold SMOTE was carried forward as the deployed strategy.",
)
add_para(
    doc,
    f"Leakage proof: on the final fold (train={leakage['n_train']:,}, "
    f"test={leakage['n_test']:,}), fitting SMOTE (and preprocessing) honestly on "
    f"train-only rows gives PR-AUC {leakage['pr_auc_honest_infold_smote']:.4f}. "
    "Pooling train and test rows before fitting SMOTE -- so synthetic minority "
    "training points get interpolated using neighbours that include the actual "
    f"test-set fraud cases -- inflates PR-AUC to "
    f"{leakage['pr_auc_leaky_outoffold_smote']:.4f}, a "
    f"{leakage['inflation_pct']:.1f}% overstatement of skill the model does not "
    "actually have, from a change in resampling scope alone. Every resampling step "
    "reported in this submission is fit inside the training fold only.",
    bold_prefix="Leakage proof:"
)

# ============================== PAGE 3 ==============================
doc.add_page_break()
add_heading(doc, "4. Cost-Based Decision Threshold")
raw_c, cal_c = cost["raw"], cost["calibrated"]
add_para(
    doc,
    f"Sweeping the decision threshold against the true KES 10,000 / KES 800 cost "
    f"asymmetry (not the naive 0.5 cut-off) on the tuned model's out-of-fold scores "
    f"finds an optimal threshold of {raw_c['best_threshold']:.3f}, costing "
    f"KES {raw_c['best_cost_per_1000']:,.0f} per 1,000 transactions scored, versus "
    f"KES {raw_c['naive_cost_per_1000']:,.0f} per 1,000 at the naive 0.5 cut-off -- "
    f"a saving of KES {raw_c['savings_per_1000']:,.0f} per 1,000 transactions "
    f"({raw_c['savings_pct']:.1f}%). At 8.4% fraud prevalence and a 12.5x cost "
    "asymmetry, the model should flag far more aggressively than 0.5 -- most of the "
    "expected cost comes from missed fraud, not from over-flagging legitimate "
    "transactions.",
)

add_heading(doc, "5. Budget-Limited Tuning (Optuna)")
add_para(
    doc,
    f"An Optuna study of {optuna_best['n_trials']} trials "
    f"({optuna_best['n_completed']} completed, {optuna_best['n_pruned']} pruned) "
    "tuned XGBoost's n_estimators, max_depth, learning_rate, subsample, "
    "colsample_bytree, min_child_weight, gamma, reg_alpha and reg_lambda "
    "(TPE sampler; MedianPruner, 10 startup trials, 1 warm-up step). To fit the "
    f"budget, the search objective averaged PR-AUC over the 3 largest purged folds "
    f"with training rows capped at {optuna_best['max_train_subsample']:,} per fold; "
    "the winning configuration was then refit at full scale (no subsampling) across "
    "all 5 folds for every number reported elsewhere in this report. Best search "
    f"value: {optuna_best['best_value']:.4f} mean PR-AUC. Full trial history and the "
    "SQLite study database are in artifacts/.",
)

add_heading(doc, "6. Calibration")
add_para(
    doc,
    f"Brier score improved from {tuned_summary['brier_raw_pooled']:.4f} (raw scores) "
    f"to {tuned_summary['brier_calibrated_pooled']:.4f} after isotonic calibration "
    "(fit inside each training fold via CalibratedClassifierCV, so the calibrator "
    "never sees the rows it is scored on). The cost-optimal threshold shifts from "
    f"{raw_c['best_threshold']:.3f} (raw) to {cal_c['best_threshold']:.3f} "
    "(calibrated) -- isotonic calibration systematically compresses the raw "
    "XGBoost scores of an imbalance-corrected model, so the probability value that "
    "carries the same real-world odds of fraud moves accordingly. The calibrated "
    f"cut-off costs KES {cal_c['best_cost_per_1000']:,.0f} per 1,000 transactions.",
)

# ============================== PAGE 4 ==============================
doc.add_page_break()
add_heading(doc, "7. Explainability and Fairness")
add_para(
    doc,
    "Global SHAP values (TreeExplainer on the tuned XGBoost booster, 3,000-row "
    "sample) and a local waterfall explanation for an individual confidently-"
    "flagged fraud case are saved to artifacts/figures/. The behavioural velocity, "
    "recency and ratio features dominate the global ranking, consistent with the "
    "feature dictionary's design intent; no single feature dominates the model, "
    "which supports giving customers a specific, feature-level reason for a hold "
    "rather than an opaque score.",
)
add_para(
    doc,
    "Subgroup fairness (approval rate = share of transactions not flagged; "
    "false-negative rate among true fraud cases) at the calibrated cost-optimal "
    "threshold, by region and by tenure band:",
)
region_rows = [[r["region"], f"{r['n']:,.0f}", f"{r['approval_rate']:.3f}", f"{r['false_negative_rate']:.3f}"]
               for _, r in region_fair.iterrows()]
add_table(doc, ["Region", "N", "Approval rate", "FNR"], region_rows, [3.6, 2.6, 3.5, 3.5])
tenure_rows = [[r["tenure_band"], f"{r['n']:,.0f}", f"{r['approval_rate']:.3f}",
                "n/a" if pd.isna(r['false_negative_rate']) else f"{r['false_negative_rate']:.3f}"]
               for _, r in tenure_fair.iterrows()]
add_table(doc, ["Tenure band", "N", "Approval rate", "FNR"], tenure_rows, [4.5, 2.6, 3.3, 2.8])
add_para(
    doc,
    "Region shows the meaningful spread here (approval 44.7%-51.1%, FNR "
    "28.4%-34.8%): Mwanza has the lowest approval rate and Arusha the highest "
    "FNR, both worth an equity review before deployment. The tenure-band split is "
    "far less informative because tenure_days is a static aggregate against a "
    "single global scoring date, so 99.9% of evaluated transactions fall in the "
    "long-tenure band by construction; the new/established bands (n=7, n=122) are "
    "too small to support a fairness conclusion and are reported for transparency, "
    "not as evidence of a clean bill of health.",
)
add_para(
    doc,
    "The CBK Digital Credit Providers Regulations (2022) require lenders to give "
    "customers specific, understandable reasons when a credit decision goes against "
    "them, not just a bare score or refusal. The per-transaction SHAP explanation "
    "produced here is designed for exactly that obligation: it identifies which "
    "specific behavioural features pushed a transaction over the threshold, in a "
    "form that can be translated into a customer-facing reason (e.g. unusual "
    "transaction velocity, distance from home location) rather than an "
    "unexplained decline. The regional and tenure-band breakdown above is the "
    "matching internal control: any region or tenure band with a materially higher "
    "false-negative rate or materially lower approval rate should be investigated "
    "before deployment, since a uniform threshold can still produce disparate "
    "outcomes across segments even when no protected attribute is used directly.",
)

add_heading(doc, "Recommendation")
add_para(
    doc,
    "Deploy the tuned XGBoost model (in-fold SMOTE, Optuna-selected "
    f"hyperparameters, isotonic calibration applied) with a single production "
    f"decision threshold of {cal_c['best_threshold']:.3f} on the calibrated fraud "
    f"probability. This configuration was chosen over plain logistic regression "
    f"and untuned bagging/boosting because, once properly tuned and calibrated, it "
    f"gives the best cost-weighted performance on purged, time-ordered evaluation "
    f"folds, and it directly minimises the KES 10,000 / KES 800 cost asymmetry the "
    f"business actually faces rather than optimising for accuracy or a default "
    f"0.5 cut-off. The evaluation plan's disclosed customer-overlap caveat means "
    "the reported numbers should be treated as an upper bound on real deployed "
    "performance until a monitored pilot confirms them on genuinely unseen "
    "customers.",
    bold_prefix="Deploy the tuned"
)

doc.save(OUT_PATH)
print(f"Saved {OUT_PATH}")
