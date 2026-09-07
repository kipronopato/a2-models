"""
Rebuilds the exact Assignment-1 cleaned modelling table.

This is a faithful re-implementation of the cleaning / feature-engineering cells in
``notebook/assignment1_report.ipynb``. Nothing here changes A1's logic, columns,
feature definitions or the excluded leak columns (manual_review_score,
settlement_status) -- per the A2 brief ("Use the output from Assignment 1, exactly
as it was. Don't change it."). The only addition is that we keep and sort by ``ts``
so Assignment 2 can build a genuine time-ordered validation plan on top of it.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = [
    PROJECT_ROOT / "Data" / "mobile_money_statements.csv",
    PROJECT_ROOT.parent / "Data" / "mobile_money_statements.csv",
]
DATA_PATH = next((path for path in DATA_CANDIDATES if path.exists()), DATA_CANDIDATES[0])
SCORING_TS = pd.Timestamp("2026-08-01 00:00:00")

NUMERICAL_COLS = [
    "recency_days", "frequency", "frequency_90d", "monetary_out", "monetary_in",
    "monetary_out_90d", "monetary_in_90d", "tenure_days",
    "cashout_ratio", "out_in_ratio", "avg_txn_size", "balance_volatility",
    "txn_velocity_7v90",
    "amount_abs", "log_amt", "hr_sin", "hr_cos", "dow_sin", "dow_cos", "payday_dist", "night_txn_ratio",
    "n_counterparty", "n_devices", "n_agents",
    "gps_missing", "gps_missing_rate", "avg_gps_dist_km",
]
CATEGORY_COLS = ["txn_type", "region", "segment"]
LEAK_NUM = ["manual_review_score"]
LEAK_CAT = ["settlement_status"]
TARGET = "is_fraud"


def _parse_amount(s):
    s = str(s).strip()
    neg = s.startswith("(") or s.startswith("-") or "Dr" in s
    digits = re.sub(r"[^0-9]", "", s)
    if digits == "":
        return np.nan
    v = float(digits)
    return -v if neg else v


def load_a1_table(data_path: Path = DATA_PATH) -> pd.DataFrame:
    raw = pd.read_csv(data_path)
    df = raw.drop_duplicates().reset_index(drop=True)

    df["amount_signed"] = df["amount"].map(_parse_amount)
    df["amount_abs"] = df["amount_signed"].abs()

    s = df["txn_time"].astype(str)
    ts = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    mask = ts.isna()
    ts.loc[mask] = pd.to_datetime(s[mask], format="%d/%m/%Y %H:%M", errors="coerce")
    mask = ts.isna()
    ts.loc[mask] = pd.to_datetime(s[mask], format="%b %d, %Y %I:%M %p", errors="coerce")
    df["ts"] = ts

    df["customer_id"] = df["reg_id"].astype(str).str.strip().str.upper()
    df = df[df["ts"] < SCORING_TS].copy()

    is_debit = df["amount_signed"] < 0
    day = df["ts"].dt.day
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek
    df["hr_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hr_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["payday_dist"] = np.minimum((day - 1).abs(), (day - 28).abs())
    df["is_night_txn"] = df["hour"].between(0, 5).astype(int)
    df["log_amt"] = np.log1p(df["amount_abs"])
    df["gps_missing"] = df["gps_lat"].isna().astype(int)

    by = df.groupby("customer_id")
    w90 = df[df["ts"] >= SCORING_TS - pd.Timedelta(days=90)]
    w7 = df[df["ts"] >= SCORING_TS - pd.Timedelta(days=7)]
    by90, by7 = w90.groupby("customer_id"), w7.groupby("customer_id")

    monetary_out = df.assign(o=np.where(is_debit, df["amount_abs"], 0)).groupby("customer_id")["o"].sum()
    monetary_in = df.assign(i=np.where(~is_debit, df["amount_abs"], 0)).groupby("customer_id")["i"].sum()
    monetary_out_90d = w90.assign(o=np.where(w90["amount_signed"] < 0, w90["amount_abs"], 0)).groupby("customer_id")["o"].sum()
    monetary_in_90d = w90.assign(i=np.where(w90["amount_signed"] >= 0, w90["amount_abs"], 0)).groupby("customer_id")["i"].sum()
    cashout_ratio = df.assign(c=(df["txn_type"] == "cashout").astype(int)).groupby("customer_id")["c"].mean()

    agg = pd.DataFrame({
        "recency_days": (SCORING_TS - by["ts"].max()).dt.total_seconds() / 86400,
        "frequency": by.size(),
        "frequency_90d": by90.size(),
        "monetary_out_90d": monetary_out_90d,
        "monetary_in_90d": monetary_in_90d,
        "tenure_days": (SCORING_TS - by["ts"].min()).dt.total_seconds() / 86400,
        "monetary_out": monetary_out,
        "monetary_in": monetary_in,
        "cashout_ratio": cashout_ratio,
        "night_txn_ratio": by["is_night_txn"].mean(),
        "balance_volatility": by["balance_after"].std().fillna(0),
        "n_counterparty": by["counterparty"].nunique(),
        "n_devices": by["device_id"].nunique(),
        "n_agents": by["agent_id"].nunique(),
        "gps_missing_rate": by["gps_missing"].mean(),
    }).reset_index()

    agg["out_in_ratio"] = agg["monetary_out"] / (agg["monetary_in"] + 1.0)
    agg["avg_txn_size"] = (agg["monetary_out"] + agg["monetary_in"]) / agg["frequency"]

    freq7 = by7.size().reindex(agg["customer_id"]).fillna(0).values
    baseline_daily = agg["frequency_90d"] / 90.0
    recent_daily = freq7 / 7.0
    agg["txn_velocity_7v90"] = recent_daily / (baseline_daily + 0.01)

    df = df.merge(agg, on="customer_id", how="left")

    gps_ok = df.dropna(subset=["gps_lat", "gps_lon"])
    home = gps_ok.groupby("customer_id")[["gps_lat", "gps_lon"]].median()
    home.columns = ["home_lat", "home_lon"]
    gps_ok = gps_ok.join(home, on="customer_id")

    EARTH_RADIUS_KM = 6371.0
    lat1, lon1 = np.radians(gps_ok["gps_lat"]), np.radians(gps_ok["gps_lon"])
    lat2, lon2 = np.radians(gps_ok["home_lat"]), np.radians(gps_ok["home_lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    gps_ok["dist_km"] = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    avg_dist = gps_ok.groupby("customer_id")["dist_km"].mean().rename("avg_gps_dist_km")
    df = df.merge(avg_dist, on="customer_id", how="left")

    df = df.sort_values("ts").reset_index(drop=True)
    return df


def get_feature_frame(df: pd.DataFrame):
    X = df[NUMERICAL_COLS + CATEGORY_COLS].copy()
    y = df[TARGET].values
    return X, y


if __name__ == "__main__":
    df = load_a1_table()
    print("shape:", df.shape, "| customers:", df["customer_id"].nunique())
    print("fraud rate: {:.4f}".format(df[TARGET].mean()))
    print("ts range:", df["ts"].min(), "->", df["ts"].max())
