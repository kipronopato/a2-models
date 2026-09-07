# a2-models

DSA 8401 Applied Machine Learning, Assignment 2: **The Cost of Being Wrong**

## What this project does

This is my Assignment 2 analysis of the mobile-money fraud model from Assignment 1. The
assignment uses loan-default terms, but my data are transactions, so I kept the same idea and
translated the decisions to fraud screening:

| Assignment wording | Used here |
|---|---|
| Borrower / application | Mobile-money transaction |
| Default | Fraudulent transaction (`is_fraud = 1`) |
| Approve | Let the transaction proceed unflagged |
| Wrongly reject a good borrower | Flag / hold a legitimate transaction for review |
| Missed default (KES 10,000) | Missed fraud — false negative |
| Wrongly rejected good borrower (KES 800) | Wrongly flagged legitimate transaction — false positive |

I used the supplied KES 10,000 cost for a missed fraud and KES 800 cost for an unnecessary flag.
The data have an 8.41% fraud rate rather than the illustrative 4% rate in the brief, so I report
the observed value instead of changing it.

The notebook compares logistic regression, random forest, and XGBoost using the same time-ordered
folds. It then compares imbalance strategies, tunes XGBoost with Optuna, chooses a cost-based
threshold, checks calibration, and produces SHAP and subgroup results.

## Running it

The source CSV is not included in this repository. Before running the notebook, place the Assignment
1 file at `Data/mobile_money_statements.csv`. It is excluded by `.gitignore` because it is supplied
separately.

```bash
cd a2-models
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 notebook/assignment2_report.ipynb
python report/build_a2_report.py
```

The Optuna section is the slowest part of the run. The report builder uses the results saved in
`artifacts/` to create the Word report. The notebook covers:

1. Time-ordered evaluation with PR-AUC as the main metric.
2. A comparison of three model families on the same folds.
3. Class weighting versus in-fold SMOTE, including a leakage check.
4. A 65-trial Optuna search with pruning.
5. Calibration, cost-based threshold selection, and reliability curves.
6. Global and local SHAP explanations and subgroup checks.
7. A final model and threshold recommendation.

The notebook writes its tables and figures to `artifacts/`, along with the Optuna database, OOF
probabilities, and saved pipeline. The reusable code is in `src/`: `data.py` handles the A1 feature
table, `cv.py` contains the time-based splitter, `models.py` builds the pipelines, and `costs.py`
handles the threshold calculations.

Only the files needed for this Assignment 2 submission are included here. Earlier Assignment 1
notebooks and duplicate working files are kept outside this folder.

## Report

`report/` contains the four-page write-up and the script used to build it from the saved artifacts.

## Repository layout

```
a2-models/
├── notebook/
│   ├── assignment2_report.ipynb   # the analysis: run this top to bottom
│   └── _build_notebook.py         # rebuilds the notebook from its cell definitions
├── src/                           # reusable data, CV, model, and cost code
├── artifacts/                     # tables, figures, model outputs, and the Optuna study
├── report/
│   ├── A2_Report.docx             # the 4-page submission
│   └── build_a2_report.py         # rebuilds the .docx from artifacts/
└── requirements.txt
```
