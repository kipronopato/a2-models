"""
Purged / blocked time-series cross-validation.

Design (expanding window):
  * The frame is sorted by transaction time ``ts``.
  * Fold k trains on every transaction up to a cutoff date, and tests on the
    following block of transactions -- so the model is always evaluated on data
    strictly after everything it was trained on. No fold ever "peeks" at the future.
  * An embargo gap (``embargo_days``) is purged immediately before each test block:
    any training row inside that gap is dropped. This removes the short-window
    autocorrelation/overlap that a plain chronological split would otherwise leave
    at the train/test border (the "purged" part of purged CV, Lopez de Prado 2018).

Known, disclosed limitation (see the report): several A1 features (recency_days,
frequency_90d, tenure_days, cashout_ratio, ...) are *customer-level* aggregates
computed once from each customer's entire history up to a single global
SCORING_TS, not recomputed at each transaction's own timestamp. Because almost
every customer transacts throughout the whole 18-month window, the same
customer's rows in an "earlier" train block and a "later" test block carry an
identical feature vector -- i.e. some future-derived information about a customer
is already baked into their earlier rows. Purging by time cannot fix this without
re-deriving point-in-time features, which would violate the "use A1's output
exactly as it was" constraint. We instead measure and disclose the customer
overlap between train and test per fold (see ``fold_diagnostics``) rather than
silently presenting the CV as fully leak-free.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PurgedBlockedTimeSeriesSplit:
    n_splits: int = 5
    embargo_days: float = 3.0
    min_train_frac: float = 0.4

    def split(self, ts: pd.Series):
        ts = pd.Series(ts).reset_index(drop=True)
        order = np.argsort(ts.values, kind="mergesort")
        n = len(ts)
        start_idx = int(n * self.min_train_frac)
        test_block_edges = np.linspace(start_idx, n, self.n_splits + 1).astype(int)

        for k in range(self.n_splits):
            test_start, test_end = test_block_edges[k], test_block_edges[k + 1]
            test_pos = order[test_start:test_end]
            train_pos_all = order[:test_start]

            test_time_start = ts.values[test_pos].min()
            embargo_cutoff = test_time_start - pd.Timedelta(days=self.embargo_days)
            train_times = ts.values[train_pos_all]
            keep = train_times < embargo_cutoff
            train_pos = train_pos_all[keep]

            yield train_pos, test_pos

    def fold_diagnostics(self, ts: pd.Series, customer_id: pd.Series):
        ts = pd.Series(ts).reset_index(drop=True)
        cust = pd.Series(customer_id).reset_index(drop=True)
        rows = []
        for k, (tr, te) in enumerate(self.split(ts)):
            train_customers = set(cust.values[tr])
            test_customers = set(cust.values[te])
            overlap = len(train_customers & test_customers)
            rows.append({
                "fold": k,
                "train_rows": len(tr),
                "test_rows": len(te),
                "train_start": ts.values[tr].min() if len(tr) else None,
                "train_end": ts.values[tr].max() if len(tr) else None,
                "test_start": ts.values[te].min(),
                "test_end": ts.values[te].max(),
                "train_customers": len(train_customers),
                "test_customers": len(test_customers),
                "customer_overlap_pct": 100.0 * overlap / max(len(test_customers), 1),
            })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.data import load_a1_table

    df = load_a1_table()
    cv = PurgedBlockedTimeSeriesSplit(n_splits=5, embargo_days=3.0)
    diag = cv.fold_diagnostics(df["ts"], df["customer_id"])
    pd.set_option("display.width", 160)
    print(diag)
