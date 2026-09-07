"""Cost-based decisioning helpers.

Cost values are the ones given in the A2 brief:
  * a missed default (here: a fraudulent transaction let through -- false negative)
    costs KES 10,000
  * a wrongly rejected good borrower (here: a legitimate transaction flagged /
    held for review -- false positive) costs KES 800
"""
import numpy as np
import pandas as pd

COST_FN = 10_000.0  # missed default / missed fraud
COST_FP = 800.0     # wrongly rejected good borrower / wrongly flagged legitimate txn


def cost_at_threshold(y_true, y_prob, threshold, cost_fn=COST_FN, cost_fp=COST_FP):
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    total_cost = fn * cost_fn + fp * cost_fp
    n = len(y_true)
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total_cost": total_cost, "cost_per_1000": total_cost / n * 1000,
        "n": n,
    }


def cost_curve(y_true, y_prob, thresholds=None, cost_fn=COST_FN, cost_fp=COST_FP):
    if thresholds is None:
        thresholds = np.linspace(0.001, 0.999, 499)
    rows = [cost_at_threshold(y_true, y_prob, t, cost_fn, cost_fp) for t in thresholds]
    return pd.DataFrame(rows)


def find_optimal_threshold(y_true, y_prob, thresholds=None, cost_fn=COST_FN, cost_fp=COST_FP):
    curve = cost_curve(y_true, y_prob, thresholds, cost_fn, cost_fp)
    best = curve.loc[curve["total_cost"].idxmin()]
    return best, curve


def savings_vs_naive(y_true, y_prob, naive_threshold=0.5, cost_fn=COST_FN, cost_fp=COST_FP):
    best, curve = find_optimal_threshold(y_true, y_prob, cost_fn=cost_fn, cost_fp=cost_fp)
    naive = cost_at_threshold(y_true, y_prob, naive_threshold, cost_fn, cost_fp)
    savings_per_1000 = naive["cost_per_1000"] - best["cost_per_1000"]
    return {
        "best_threshold": float(best["threshold"]),
        "best_cost_per_1000": float(best["cost_per_1000"]),
        "naive_threshold": naive_threshold,
        "naive_cost_per_1000": float(naive["cost_per_1000"]),
        "savings_per_1000": float(savings_per_1000),
        "savings_pct": float(100 * savings_per_1000 / naive["cost_per_1000"]) if naive["cost_per_1000"] else 0.0,
        "best_row": best,
        "naive_row": naive,
    }
