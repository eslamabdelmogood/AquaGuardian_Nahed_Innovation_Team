"""
validate_against_real_data_v2.py
===================================
Improves on validate_against_real_data.py by fixing the core problem
that made the first version's rule look bad: it was evaluated the same
way it was designed (a fixed threshold=70, chosen by hand, tested on
100% of the data) -- which is not a fair test, and doesn't tell you
whether a *better-tuned* rule could do meaningfully better.

This version:
  1. Splits the year CHRONOLOGICALLY into a train period and a held-out
     test period (default: first 8 months train, last 4 months test),
     with a 7-day gap at the boundary to reduce label leakage from the
     +/-7-day fault_d7 window smearing across the split point.
  2. On the TRAIN period only: tunes a per-column threshold to maximize
     F1, and separately fits a simple logistic regression combining the
     top columns (median imputation from train, never from test).
  3. Evaluates every candidate on the TEST period only -- data it never
     saw during tuning -- and reports it next to the naive "always
     flag" and "never flag" baselines computed on that SAME test period.

If a tuned rule cannot beat "always flag" on the held-out test period,
that is reported plainly, not hidden -- the point of this script is an
honest answer, not a better-looking number.

Usage:
  python3 validate_against_real_data_v2.py dataset.csv
  python3 validate_against_real_data_v2.py dataset.csv --train-months 8
"""
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

PREDICTOR_COLUMNS = [
    "4700008966_kW", "4700008966_power_hour", "4408411600_kW", "4408411600_power_hour",
    "4608927500_kvar_hour",
    "238045_ws_temp", "279625_ws_temp", "279804_ws_level", "279805_ws_vigor",
    "279623_ws_vigor", "319235_ws_temp", "279831_ws_temp", "326417_ws_temp", "326416_ws_vigor",
    "temp_site1_anomaly_score", "sfc_temp_site1_anomaly_score",
    "gw_lvl_site2_anomaly_score", "gw_temp_site2_anomaly_score",
]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "fault_d7" not in df.columns:
        raise ValueError("Expected column 'fault_d7' not found.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame, train_months: int, gap_days: int = 7):
    start = df["timestamp"].min()
    train_end = start + pd.DateOffset(months=train_months)
    test_start = train_end + pd.Timedelta(days=gap_days)
    train = df[df["timestamp"] < train_end].copy()
    test = df[df["timestamp"] >= test_start].copy()
    return train, test


def report(name, y_true, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"  {name:38s}  precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}")
    return f1


def tune_single_column_threshold(train_col, train_y):
    """Sweep thresholds on TRAIN data only, pick the one that maximizes F1."""
    candidates = np.percentile(train_col.dropna(), np.arange(5, 100, 5))
    best_t, best_f1 = None, -1
    for t in candidates:
        pred = (train_col > t).astype(int)
        pred = pred.reindex(train_y.index).fillna(0)
        f1 = f1_score(train_y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def main(path: str, train_months: int):
    df = load(path)
    print(f"Loaded {len(df)} rows spanning {df['timestamp'].min()} to {df['timestamp'].max()}\n")

    train, test = chronological_split(df, train_months)
    print(f"Train period: {len(train)} rows ({train['timestamp'].min()} to {train['timestamp'].max()})")
    print(f"Test period:  {len(test)} rows ({test['timestamp'].min()} to {test['timestamp'].max()})")
    print(f"Train positive rate: {train['fault_d7'].mean():.1%}  |  Test positive rate: {test['fault_d7'].mean():.1%}\n")

    # --- Rank columns by AUC on TRAIN only ---
    aucs = {}
    for col in PREDICTOR_COLUMNS:
        sub = train[[col, "fault_d7"]].dropna()
        if sub["fault_d7"].nunique() < 2 or len(sub) < 50:
            continue
        try:
            aucs[col] = roc_auc_score(sub["fault_d7"], sub[col])
        except ValueError:
            continue
    ranked = sorted(aucs.items(), key=lambda kv: kv[1], reverse=True)
    print("Top columns by AUC, computed on TRAIN only (not test):")
    for col, auc in ranked[:6]:
        print(f"  {col:32s} AUC={auc:.3f}")
    print()

    top_col_name = ranked[0][0] if ranked else None
    top5_cols = [c for c, _ in ranked[:5]]

    results = {}
    y_test = test["fault_d7"].astype(int)

    # --- Baselines, computed on TEST period ---
    print("=== Baselines (test period) ===")
    results["always_flag"] = report("Naive: always flag", y_test, np.ones(len(y_test), dtype=int))
    results["never_flag"] = report("Naive: never flag", y_test, np.zeros(len(y_test), dtype=int))
    print()

    # --- Candidate 1: single best column, threshold TUNED ON TRAIN ---
    if top_col_name:
        train_sub = train[[top_col_name, "fault_d7"]].dropna()
        best_t, train_f1 = tune_single_column_threshold(train_sub[top_col_name], train_sub["fault_d7"])
        test_sub = test[[top_col_name, "fault_d7"]].dropna()
        pred = (test_sub[top_col_name] > best_t).astype(int)
        print(f"=== Candidate 1: single column '{top_col_name}', threshold tuned on TRAIN = {best_t:.1f} ===")
        results["tuned_single_column"] = report(f"Tuned threshold (test, n={len(test_sub)})", test_sub["fault_d7"], pred)
        print()

    # --- Candidate 2: logistic regression combining top-5 columns ---
    if len(top5_cols) >= 2:
        train_X = train[top5_cols].copy()
        medians = train_X.median()
        train_X = train_X.fillna(medians)
        train_y = train["fault_d7"].astype(int)

        test_X = test[top5_cols].copy().fillna(medians)  # impute with TRAIN medians, never test medians
        test_y = y_test

        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(train_X, train_y)
        pred = clf.predict(test_X)
        print(f"=== Candidate 2: logistic regression on {top5_cols} (class_weight='balanced') ===")
        results["logistic_regression"] = report(f"Logistic regression (test, n={len(test_X)})", test_y, pred)
        print("  Coefficients:", dict(zip(top5_cols, np.round(clf.coef_[0], 4))))
        print()

    print("=== Summary: does anything beat naive 'always flag' on held-out F1? ===")
    baseline_f1 = results["always_flag"]
    for name, f1 in results.items():
        if name == "always_flag":
            continue
        verdict = "BEATS baseline" if f1 > baseline_f1 else "does NOT beat baseline"
        print(f"  {name:22s} f1={f1:.3f} vs baseline f1={baseline_f1:.3f}  -> {verdict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--train-months", type=int, default=8)
    args = parser.parse_args()
    main(args.csv_path, args.train_months)
