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

The modelling data is included at `Data/mobile_money_statements.csv`.

```bash
cd a2-models
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 notebook/assignment2_report.ipynb
python report/build_a2_report.py
```

The notebook covers time-ordered evaluation, three model families, imbalance handling, a 65-trial
Optuna search, calibration, cost-based threshold selection, SHAP explanations, and subgroup checks.
The Optuna section is the slowest part of the run. Results are written to `artifacts/`.

## Report

`report/` contains the four-page write-up and the script used to build it.

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
