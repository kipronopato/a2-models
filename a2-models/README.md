# a2-models — DSA 8401 Assignment 2: The Cost of Being Wrong

## Scope note (read this first)

Assignment 2 asks us to build on "the output from Assignment 1, exactly as it was."
A1's deliverable is a **mobile-money fraud-scoring pipeline** (target `is_fraud`,
base rate 8.41%), not a loan-approval/default dataset. Rather than build a second,
unrelated dataset, this submission keeps A1's cleaning and 18 engineered features
completely unchanged and maps the assignment's credit-risk vocabulary onto the
fraud-scoring task one-for-one, as the brief explicitly allows ("use the instructor's
reference pipeline instead, no penalty... just say clearly you did this" — here we
went one step further and kept our own A1 pipeline, disclosed and mapped):

| Assignment 2 term | This submission |
|---|---|
| Borrower / application | Mobile-money transaction |
| Default | Fraudulent transaction (`is_fraud = 1`) |
| Approve | Let the transaction proceed unflagged |
| Wrongly reject a good borrower | Flag / hold a legitimate transaction for review |
| Missed default (KES 10,000) | Missed fraud — false negative |
| Wrongly rejected good borrower (KES 800) | Wrongly flagged legitimate transaction — false positive |

The cost values (10,000 / 800, a 12.5x asymmetry) are used exactly as given in the
brief. The observed base rate is 8.41%, not 4% — reported honestly rather than forced
to match, since the same "rare, costly, asymmetric" logic the assignment is testing
applies regardless of the exact rate.

## Reproducing this submission

The modelling input is not committed to this repository. Place the Assignment 1 source file at
`Data/mobile_money_statements.csv` before executing the notebook. In the original workspace the
loader also accepts the existing parent-level `Data/` folder, but the project-local path is the
recommended layout for a standalone clone. The raw file is excluded by `.gitignore`.

```bash
cd a2-models
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 notebook/assignment2_report.ipynb   # ~25-30 min, mostly the Optuna study
python report/build_a2_report.py   # rebuilds A2_Report.docx from the artifacts the notebook wrote
```

Or open `notebook/assignment2_report.ipynb` in Jupyter/JupyterLab and Run All — every cell
executes top to bottom with no manual steps in between. The notebook covers, in order:

1. **Evaluation plan** — purged/blocked time-series CV, PR-AUC vs. ROC-AUC rationale.
2. **Ladder of models** — logistic regression vs. random forest vs. XGBoost, identical folds.
3. **Imbalance handling** — class weighting vs. in-fold SMOTE, plus the required proof that
   resampling outside the folds inflates the score.
4. **Budget-limited tuning** — a 65-trial Optuna study (SQLite storage at
   `artifacts/optuna_study.db`), MedianPruner, subsampled search folds.
5. **Final model + cost threshold + calibration** — full-scale refit with the tuned
   hyperparameters, cost-based threshold, reliability diagram, Brier score before/after,
   threshold recalculated on calibrated scores.
6. **Explainability and fairness** — global/local SHAP, subgroup fairness (region, tenure
   band), saves the final production pipeline.
7. **Recommendation** — one model, one threshold.

All outputs land in `artifacts/` (`tables/`, `figures/`, `oof/`, the Optuna DB, and
`final_fraud_pipeline.joblib`). `src/` holds the shared, reusable modules the notebook imports:
`data.py` (A1's cleaning/features, untouched), `cv.py` (the purged/blocked splitter),
`models.py` (pipelines), `costs.py` (cost-curve helpers) — kept out of the notebook so the
notebook reads as analysis narrative rather than plumbing.

The repository intentionally contains only the A2 analysis, its generated evidence, and the
reproduction code. Earlier Assignment 1 notebooks, duplicate data folders, intermediate reports,
and local model files are outside this submission folder and are not needed here.

## Report

`report/` contains the 4-page write-up (`A2_Report.docx`) and its build script
(`build_a2_report.py`), which reads every number straight from `artifacts/` so it stays in
sync with whatever the notebook last produced.

## Repository layout

```
a2-models/
├── notebook/
│   ├── assignment2_report.ipynb   # the analysis: run this top to bottom
│   └── _build_notebook.py         # (dev tool) regenerates the .ipynb cells from source; not part of the analysis
├── src/                           # small reusable library the notebook imports
├── artifacts/                     # everything the notebook writes: tables/, figures/, oof/, the Optuna DB, the saved pipeline
├── report/
│   ├── A2_Report.docx             # the 4-page submission
│   └── build_a2_report.py         # rebuilds the .docx from artifacts/
└── requirements.txt
```
