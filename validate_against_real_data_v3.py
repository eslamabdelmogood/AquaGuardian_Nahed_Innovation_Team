"""
validate_against_real_data_v3.py
===================================
Addresses every gap in v2, per reviewer feedback:

  1. Confusion matrix for the logistic regression model.
  2. Precision/Recall/F1 at a sweep of thresholds (not just one).
  3. ROC-AUC AND PR-AUC. PR-AUC matters here for a specific, worth-stating
     reason: fault_d7 is majority-POSITIVE (~74-85%), so the class that
     is actually rare is NEGATIVE ("not near a leak"). The no-skill
     PR-AUC baseline for the positive class equals the positive
     prevalence itself (~0.85) -- so a positive-class PR-AUC that merely
     looks high (e.g. 0.85) may be exactly at chance. This script prints
     that no-skill reference line next to the model's real PR-AUC so
     "high" and "beats chance" can't be confused with each other. It
     also reports PR-AUC for the NEGATIVE class, since that is the
     actually-imbalanced direction in this dataset.
  4. Three fairer baselines, not just "always flag":
       - Historical prevalence  (constant probability = train prevalence)
       - Previous-hour persistence (predict fault_d7[t] = fault_d7[t-1])
       - Simple single-signal rule (best single column, MCC-tuned on train)
  5. Walk-forward (expanding-window) time validation: monthly folds
     across the whole year, not one arbitrary 8/4 split. Column
     selection AND threshold tuning happen inside each fold's training
     data only -- never using that fold's own test data, and never
     using a later fold's data.

Usage:
  python3 validate_against_real_data_v3.py dataset.csv
  python3 validate_against_real_data_v3.py dataset.csv --min-train-months 3 --top-k 5
"""
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score, confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)

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
    df["fault_d7"] = df["fault_d7"].astype(int)
    return df


def walk_forward_folds(df, min_train_months, gap_days=7):
    """Expanding-window monthly folds: fold k trains on everything before
    month k (minus a gap), tests on month k itself. Yields (label, train, test)."""
    start = df["timestamp"].min()
    end = df["timestamp"].max()
    month_starts = pd.date_range(start.replace(day=1), end, freq="MS")
    folds = []
    for i in range(min_train_months, len(month_starts)):
        test_month_start = month_starts[i]
        test_month_end = month_starts[i + 1] if i + 1 < len(month_starts) else end + pd.Timedelta(hours=1)
        train_cutoff = test_month_start - pd.Timedelta(days=gap_days)
        train = df[df["timestamp"] < train_cutoff]
        test = df[(df["timestamp"] >= test_month_start) & (df["timestamp"] < test_month_end)]
        if len(train) < 200 or len(test) < 50:
            continue
        if train["fault_d7"].nunique() < 2 or test["fault_d7"].nunique() < 2:
            continue  # can't evaluate a fold with only one class present
        folds.append((test_month_start.strftime("%Y-%m"), train, test))
    return folds


def select_top_columns(train_df, k):
    aucs = {}
    for col in PREDICTOR_COLUMNS:
        sub = train_df[[col, "fault_d7"]].dropna()
        if sub["fault_d7"].nunique() < 2 or len(sub) < 50:
            continue
        try:
            aucs[col] = roc_auc_score(sub["fault_d7"], sub[col])
        except ValueError:
            continue
    ranked = sorted(aucs.items(), key=lambda kv: kv[1], reverse=True)
    return [c for c, _ in ranked[:k]], ranked


def tune_threshold_mcc(train_col, train_y):
    candidates = np.percentile(train_col.dropna(), np.arange(5, 100, 5))
    best_t, best_mcc = None, -2
    for t in candidates:
        pred = (train_col > t).astype(int).reindex(train_y.index).fillna(0)
        mcc = matthews_corrcoef(train_y, pred) if len(set(pred)) > 1 else -2
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t
    return best_t


def safe_pr_auc(y_true, scores):
    try:
        return average_precision_score(y_true, scores)
    except ValueError:
        return None


def safe_roc_auc(y_true, scores):
    try:
        return roc_auc_score(y_true, scores)
    except ValueError:
        return None


def evaluate_fold(label, train, test, top_k):
    top_cols, ranked = select_top_columns(train, top_k)
    if not top_cols:
        return None
    top_col = top_cols[0]
    prevalence = train["fault_d7"].mean()

    # ---- Logistic regression model ----
    train_X = train[top_cols].copy()
    medians = train_X.median()
    train_X = train_X.fillna(medians)
    train_y = train["fault_d7"].astype(int)
    test_X = test[top_cols].copy().fillna(medians)
    test_y = test["fault_d7"].astype(int)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(train_X, train_y)
    proba = clf.predict_proba(test_X)[:, 1]
    pred_50 = (proba >= 0.5).astype(int)

    model = {
        "roc_auc": safe_roc_auc(test_y, proba),
        "pr_auc_pos": safe_pr_auc(test_y, proba),
        "pr_auc_neg": safe_pr_auc(1 - test_y, 1 - proba),
        "mcc": matthews_corrcoef(test_y, pred_50) if len(set(pred_50)) > 1 else 0.0,
        "f1": f1_score(test_y, pred_50, zero_division=0),
        "bal_acc": balanced_accuracy_score(test_y, pred_50),
        "confusion": confusion_matrix(test_y, pred_50, labels=[0, 1]),
        "proba": proba, "y_true": test_y.values,
        "top_cols": top_cols,
    }

    # ---- Baseline 1: historical prevalence (constant probability) ----
    const_proba = np.full(len(test_y), prevalence)
    prevalence_baseline = {
        "roc_auc": 0.5,  # a constant score has no ranking power by definition
        "pr_auc_pos": safe_pr_auc(test_y, const_proba),  # equals test prevalence, this IS the no-skill line
        "pr_auc_neg": safe_pr_auc(1 - test_y, 1 - const_proba),
        "mcc": 0.0,
        "f1": f1_score(test_y, np.ones(len(test_y)) if prevalence >= 0.5 else np.zeros(len(test_y)), zero_division=0),
    }

    # ---- Baseline 2: previous-hour persistence ----
    test_sorted = test.sort_values("timestamp")
    prev_pred = test_sorted["fault_d7"].shift(1)
    # first row of the fold has no in-fold previous hour; carry the last train label in
    if len(train) > 0:
        prev_pred.iloc[0] = train.sort_values("timestamp")["fault_d7"].iloc[-1]
    prev_pred = prev_pred.bfill().astype(int)
    y_aligned = test_sorted["fault_d7"].astype(int)
    persistence_baseline = {
        "mcc": matthews_corrcoef(y_aligned, prev_pred) if len(set(prev_pred)) > 1 else 0.0,
        "f1": f1_score(y_aligned, prev_pred, zero_division=0),
        "bal_acc": balanced_accuracy_score(y_aligned, prev_pred),
    }

    # ---- Baseline 3: simple single-signal rule (MCC-tuned threshold on train) ----
    train_sub = train[[top_col, "fault_d7"]].dropna()
    best_t = tune_threshold_mcc(train_sub[top_col], train_sub["fault_d7"])
    test_sub = test[[top_col, "fault_d7"]].dropna()
    single_pred = (test_sub[top_col] > best_t).astype(int) if best_t is not None else pd.Series(0, index=test_sub.index)
    single_rule_baseline = {
        "mcc": matthews_corrcoef(test_sub["fault_d7"], single_pred) if len(set(single_pred)) > 1 else 0.0,
        "f1": f1_score(test_sub["fault_d7"], single_pred, zero_division=0),
        "column": top_col, "threshold": best_t,
    }

    return {
        "label": label, "n_train": len(train), "n_test": len(test),
        "train_prevalence": prevalence, "test_prevalence": test_y.mean(),
        "model": model, "prevalence_baseline": prevalence_baseline,
        "persistence_baseline": persistence_baseline, "single_rule_baseline": single_rule_baseline,
    }


def print_confusion_matrix(cm):
    print("           Predicted 0   Predicted 1")
    print(f"  Actual 0    {cm[0][0]:6d}        {cm[0][1]:6d}")
    print(f"  Actual 1    {cm[1][0]:6d}        {cm[1][1]:6d}")


def print_threshold_sweep(y_true, proba):
    print("  threshold   precision   recall   f1")
    for t in np.arange(0.1, 0.95, 0.1):
        pred = (proba >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        print(f"    {t:.2f}        {p:.3f}      {r:.3f}   {f1:.3f}")


def main(path, min_train_months, top_k):
    df = load(path)
    print(f"Loaded {len(df)} rows, {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"Overall fault_d7 prevalence: {df['fault_d7'].mean():.1%}\n")

    folds = walk_forward_folds(df, min_train_months)
    print(f"Built {len(folds)} walk-forward monthly folds (expanding window, {min_train_months}-month minimum, 7-day gap)\n")

    fold_results = []
    for label, train, test in folds:
        r = evaluate_fold(label, train, test, top_k)
        if r is not None:
            fold_results.append(r)

    if not fold_results:
        print("No valid folds could be evaluated (check data density / class balance per month).")
        return

    print("=" * 100)
    print(f"{'Fold':10s} {'n_test':>8s} {'test_prev':>10s} | {'Model':>28s} | {'Prevalence BL':>16s} | {'Persistence BL':>16s} | {'Single-signal BL':>18s}")
    print(f"{'':10s} {'':>8s} {'':>10s} | {'ROC-AUC':>9s}{'PR-AUC+':>9s}{'MCC':>9s} | {'PR-AUC+':>8s}{'MCC':>8s} | {'MCC':>8s}{'F1':>8s} | {'MCC':>9s}{'F1':>9s}")
    print("-" * 100)
    for r in fold_results:
        m = r["model"]
        pb = r["prevalence_baseline"]
        pe = r["persistence_baseline"]
        sr = r["single_rule_baseline"]
        print(f"{r['label']:10s} {r['n_test']:8d} {r['test_prevalence']:9.1%} | "
              f"{m['roc_auc']:9.3f}{m['pr_auc_pos']:9.3f}{m['mcc']:+9.3f} | "
              f"{pb['pr_auc_pos']:8.3f}{pb['mcc']:+8.3f} | "
              f"{pe['mcc']:+8.3f}{pe['f1']:8.3f} | "
              f"{sr['mcc']:+9.3f}{sr['f1']:9.3f}")
    print("=" * 100)

    def agg(key_path):
        vals = []
        for r in fold_results:
            d = r
            for k in key_path:
                d = d[k]
            if d is not None:
                vals.append(d)
        return (np.mean(vals), np.std(vals)) if vals else (None, None)

    print("\n=== Aggregate across all folds (mean +/- std) ===")
    for name, path_ in [
        ("Model MCC", ("model", "mcc")),
        ("Model ROC-AUC", ("model", "roc_auc")),
        ("Model PR-AUC (positive class)", ("model", "pr_auc_pos")),
        ("Model PR-AUC (negative class -- the rarer direction here)", ("model", "pr_auc_neg")),
        ("Prevalence-baseline PR-AUC (no-skill reference line)", ("prevalence_baseline", "pr_auc_pos")),
        ("Persistence-baseline MCC", ("persistence_baseline", "mcc")),
        ("Single-signal-rule MCC", ("single_rule_baseline", "mcc")),
    ]:
        mean, std = agg(path_)
        if mean is not None:
            print(f"  {name:58s} {mean:+.3f} (+/- {std:.3f})")

    # --- Detail view: confusion matrix + threshold sweep for the LAST fold ---
    last = fold_results[-1]
    print(f"\n=== Detail for most recent fold ({last['label']}), model = logistic regression on {last['model']['top_cols']} ===")
    print("Confusion matrix @ threshold 0.5:")
    print_confusion_matrix(last["model"]["confusion"])
    print("\nPrecision/Recall/F1 across thresholds:")
    print_threshold_sweep(last["model"]["y_true"], last["model"]["proba"])

    print("\n=== How to read this ===")
    print("- MCC = 0.000 exactly is what EVERY constant-output baseline scores, regardless of prevalence.")
    print("  A model's MCC clearing 0 by a real margin, consistently across folds, is the actual evidence bar.")
    print("- Prevalence-baseline PR-AUC (positive class) is the no-skill reference line for PR-AUC -- if the")
    print("  model's own PR-AUC (positive) is close to this number, PR-AUC alone is not showing real skill.")
    print("- Persistence (predict-previous-hour) is often a strong baseline for slow-changing window labels")
    print("  like fault_d7 -- if the model can't beat it, that says the model isn't adding value over 'nothing")
    print("  changed since last hour', which is a meaningful, honest limitation to report as-is.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--min-train-months", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    main(args.csv_path, args.min_train_months, args.top_k)
