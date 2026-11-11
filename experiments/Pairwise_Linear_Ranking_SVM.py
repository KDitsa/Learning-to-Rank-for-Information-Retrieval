"""
Pairwise Linear Ranking SVM with:
  - query-level 80/20 train/test split
  - GroupKFold cross-validation on training queries to choose C
  - evaluation: Precision, Recall, F1, NDCG, pairwise accuracy
"""

from pathlib import Path
import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "features" / "dataset.csv"
MODEL_DIR = ROOT / "model" / "Pairwise_Linear_Ranking_SVM"

QID = "qid"
DOCID = "docid"
REL = "relevance"

EXPECTED_FEATURES = 110
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

SELECTION_K = 10            # k used to choose C
REPORT_KS = (5, 10)         # k values reported on the test set
C_VALUES = [    
    0.0001,
    0.0003,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1, 
    1, 
    10, 
    100]

MAX_PAIRS_PER_QUERY = 2000


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    rename = {}
    if QID not in df.columns and "query_id" in df.columns:
        rename["query_id"] = QID
    if DOCID not in df.columns:
        if "doc_id" in df.columns:
            rename["doc_id"] = DOCID
        elif "document_id" in df.columns:
            rename["document_id"] = DOCID
    df = df.rename(columns=rename)

    missing = {QID, DOCID, REL} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    feature_columns = [c for c in df.columns if c not in {QID, DOCID, REL}]

    if len(feature_columns) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, found "
            f"{len(feature_columns)}: {feature_columns}"
        )

    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="raise")
    df[REL] = pd.to_numeric(df[REL], errors="raise")

    if df[feature_columns].isnull().any().any():
        raise ValueError("Feature columns contain missing values. Fix the feature builder.")
    if not np.isfinite(df[feature_columns].to_numpy()).all():
        raise ValueError("Feature columns contain infinite values.")
    if df[REL].isnull().any():
        raise ValueError("Relevance column contains missing values.")
    if not np.equal(df[REL].to_numpy(), np.floor(df[REL].to_numpy())).all():
        raise ValueError("Relevance labels must be integers.")
    if (df[REL] < 0).any():
        raise ValueError("Relevance labels must be non-negative.")
    if df.duplicated(subset=[QID, DOCID]).any():
        raise ValueError("Duplicate (qid, docid) rows found.")

    df[REL] = df[REL].astype(int)
    df[QID] = df[QID].astype(str)
    df[DOCID] = df[DOCID].astype(str)
    df = df.reset_index(drop=True)

    print(f"Dataset shape: {df.shape}")
    print(f"Queries: {df[QID].nunique()}")
    print(f"Features: {len(feature_columns)}")
    print(f"Relevance labels: {sorted(df[REL].unique())}")

    return df, feature_columns


def create_pairwise_data(df, feature_columns, rng=None):
    """
    For each query and each pair (i, j) with different relevance:
        x = X_better - X_worse  -> label +1
        x = X_worse - X_better  -> label -1
    """
    df = df.reset_index(drop=True)
    X_all = df[feature_columns].to_numpy(dtype=np.float64)
    y_all = df[REL].to_numpy()

    X_parts, y_parts = [], []

    for _, idx in df.groupby(QID, sort=False).indices.items():
        if len(idx) < 2:
            continue

        Xq = X_all[idx]
        yq = y_all[idx]

        a, b = np.triu_indices(len(idx), k=1)
        keep = yq[a] != yq[b]
        a, b = a[keep], b[keep]

        if len(a) == 0:
            continue

        if MAX_PAIRS_PER_QUERY is not None and len(a) > MAX_PAIRS_PER_QUERY:
            if rng is None:
                rng = np.random.default_rng(RANDOM_STATE)
            sel = rng.choice(len(a), MAX_PAIRS_PER_QUERY, replace=False)
            a, b = a[sel], b[sel]

        sign = np.where(yq[a] > yq[b], 1.0, -1.0)
        preferred = (Xq[a] - Xq[b]) * sign[:, None]   # better minus worse

        X_parts.append(preferred)
        y_parts.append(np.ones(len(preferred), dtype=np.int32))
        X_parts.append(-preferred)
        y_parts.append(-np.ones(len(preferred), dtype=np.int32))

    if not X_parts:
        raise ValueError("No unequal-relevance pairs found; cannot train.")

    return np.vstack(X_parts), np.concatenate(y_parts)


def create_model(C):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(
            C=C,
            penalty="l2",
            loss="squared_hinge",
            dual=False,            # 22 features, many samples
            fit_intercept=False,   # pairwise differences
            tol=1e-4,
            max_iter=10000,
            random_state=RANDOM_STATE,
        )),
    ])


def fit_model(C, train_df, feature_columns):
    X_pairs, y_pairs = create_pairwise_data(train_df, feature_columns)
    model = create_model(C)
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        model.fit(X_pairs, y_pairs)
    return model


def score_documents(model, df, feature_columns):
    return model.decision_function(df[feature_columns].to_numpy(dtype=np.float64))


def per_query_metrics(df, scores, k):
    """
    Per-query P@k, R@k, F1@k, NDCG@k.
    Relevant = relevance > 0. Queries with no relevant docs are skipped.
    """
    tmp = pd.DataFrame({
        QID: df[QID].to_numpy(),
        REL: df[REL].to_numpy(),
        "score": np.asarray(scores, dtype=float),
    })

    rows = []
    for qid, g in tmp.groupby(QID, sort=False):
        labels = g.sort_values("score", ascending=False, kind="stable")[REL].to_numpy()

        n_relevant = int((labels > 0).sum())
        if n_relevant == 0:
            continue

        kk = min(k, len(labels))
        top = labels[:kk]
        hits = int((top > 0).sum())

        precision = hits / kk
        recall = hits / n_relevant
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        discounts = 1.0 / np.log2(np.arange(2, kk + 2))
        dcg = ((2.0 ** top - 1) * discounts).sum()
        ideal = np.sort(labels)[::-1][:kk]
        idcg = ((2.0 ** ideal - 1) * discounts).sum()
        ndcg = dcg / idcg if idcg > 0 else 0.0

        rows.append({
            QID: qid,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ndcg": ndcg,
        })

    return pd.DataFrame(rows)


def summarize(per_query_df):
    return {
        "precision": per_query_df["precision"].mean(),
        "recall": per_query_df["recall"].mean(),
        "f1": per_query_df["f1"].mean(),
        "ndcg": per_query_df["ndcg"].mean(),
        "n_queries": len(per_query_df),
    }


def pairwise_accuracy(df, scores):
    """Fraction of unequal-relevance pairs ordered correctly (ties = 0.5)."""
    df = df.reset_index(drop=True)
    y_all = df[REL].to_numpy()
    scores = np.asarray(scores, dtype=float)

    correct, total = 0.0, 0

    for _, idx in df.groupby(QID, sort=False).indices.items():
        if len(idx) < 2:
            continue
        yq = y_all[idx]
        sq = scores[idx]

        a, b = np.triu_indices(len(idx), k=1)
        keep = yq[a] != yq[b]
        a, b = a[keep], b[keep]
        if len(a) == 0:
            continue

        sign = np.where(yq[a] > yq[b], 1.0, -1.0)
        diff = (sq[a] - sq[b]) * sign

        correct += (diff > 0).sum() + 0.5 * (diff == 0).sum()
        total += len(a)

    return correct / total if total else float("nan")


def tune_C(train_df, feature_columns):
    n_queries = train_df[QID].nunique()
    n_splits = min(N_SPLITS, n_queries)
    if n_splits < 2:
        raise ValueError("Need at least 2 training queries for cross-validation.")

    gkf = GroupKFold(n_splits=n_splits)
    rows = []

    for C in C_VALUES:
        print(f"\nTesting C = {C}")

        for fold, (fit_idx, val_idx) in enumerate(
            gkf.split(train_df, groups=train_df[QID]), start=1
        ):
            fit_df = train_df.iloc[fit_idx].reset_index(drop=True)
            val_df = train_df.iloc[val_idx].reset_index(drop=True)

            model = fit_model(C, fit_df, feature_columns)
            scores = score_documents(model, val_df, feature_columns)

            m = summarize(per_query_metrics(val_df, scores, SELECTION_K))
            acc = pairwise_accuracy(val_df, scores)

            rows.append({
                "C": C,
                "fold": fold,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "ndcg": m["ndcg"],
                "pairwise_accuracy": acc,
            })

            print(
                f"  Fold {fold}: NDCG@{SELECTION_K}={m['ndcg']:.4f}  "
                f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
                f"F1={m['f1']:.4f}  PairAcc={acc:.4f}"
            )

    cv_df = pd.DataFrame(rows)
    summary = cv_df.drop(columns="fold").groupby("C").agg(["mean", "std"])

    mean_ndcg = cv_df.groupby("C")["ndcg"].mean()
    best_C = float(mean_ndcg.idxmax())

    print("\nCross-validation summary (mean / std over folds):")
    print(summary.round(4).to_string())
    print(f"\nBest C: {best_C}  (mean validation NDCG@{SELECTION_K} = {mean_ndcg[best_C]:.4f})")

    return best_C, cv_df, summary


def build_weight_table(model, feature_columns):
    svm = model.named_steps["svm"]
    scaler = model.named_steps["scaler"]

    standardized = svm.coef_[0]
    raw_units = standardized / scaler.scale_

    total = np.abs(standardized).sum()
    table = pd.DataFrame({
        "feature": feature_columns,
        "standardized_coefficient": standardized,
        "absolute_importance": np.abs(standardized),
        "importance_percentage": (100 * np.abs(standardized) / total) if total > 0 else 0.0,
        "raw_unit_weight": raw_units,
    }).sort_values("absolute_importance", ascending=False).reset_index(drop=True)

    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def main():
    parser = argparse.ArgumentParser(description="Tuned pairwise Ranking SVM.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df, feature_columns = load_dataset(args.data)

    # ---- query-level split (same seed as train.py) ----
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(df, groups=df[QID]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    assert set(train_df[QID]).isdisjoint(set(test_df[QID]))

    print(f"\nTraining queries: {train_df[QID].nunique()}  rows: {len(train_df)}")
    print(f"Test queries:     {test_df[QID].nunique()}  rows: {len(test_df)}")

    # ---- tune C ----
    best_C, cv_df, cv_summary = tune_C(train_df, feature_columns)
    cv_df.to_csv(MODEL_DIR / "outliers_cross_validation_results.csv", index=False)
    cv_summary.to_csv(MODEL_DIR / "outliers_cross_validation_summary.csv")

    # ---- retrain on all training queries ----
    final_model = fit_model(best_C, train_df, feature_columns)
    test_scores = score_documents(final_model, test_df, feature_columns)

    # ---- final test evaluation: Ranking SVM ----
    report_rows = []
    per_query_store = {}

    for k in REPORT_KS:
        svm_pq = per_query_metrics(test_df, test_scores, k)
        per_query_store[k] = svm_pq
        report_rows.append({
            "method": "Pairwise Linear Ranking SVM",
            "k": k,
            **summarize(svm_pq)
        })

    report_df = pd.DataFrame(report_rows)
    svm_pairwise = pairwise_accuracy(test_df, test_scores)

    print("\n================ FINAL TEST RESULTS ================")
    print(f"Chosen C: {best_C}")
    print(report_df.round(4).to_string(index=False))
    print(f"\nTest pairwise accuracy: {svm_pairwise:.4f}")

    # ---- save analysis outputs ----
    # Cross-validation results for every C and fold
    cv_df.to_csv(MODEL_DIR / "cross_validation_results.csv", index=False)

    # Cross-validation summary (mean and standard deviation)
    cv_summary.to_csv(MODEL_DIR / "cross_validation_summary.csv")

    # Final test metrics
    report_df.to_csv(MODEL_DIR / "test_metrics.csv", index=False)

    # Feature coefficient analysis
    weights = build_weight_table(final_model, feature_columns)
    weights.to_csv(MODEL_DIR / "feature_weights.csv", index=False)

    # Save trained model
    joblib.dump(final_model, MODEL_DIR / "ranking_svm_model.joblib")

    # Save model configuration
    metadata = {
        "model_type": "Pairwise Linear Ranking SVM",
        "feature_columns": feature_columns,
        "best_C": best_C,
        "c_values_tried": C_VALUES,
        "cv_folds": N_SPLITS,
        "selection_metric": f"NDCG@{SELECTION_K}",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "test_pairwise_accuracy": float(svm_pairwise),
    }

    with open(MODEL_DIR / "model_metadata.json","w",encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nTop 10 features:")
    print(
        weights[
            ["rank", "feature", "standardized_coefficient",
            "importance_percentage"]
        ].head(10).round(4).to_string(index=False)
    )

    print(f"\nAll outputs saved in: {MODEL_DIR}")

if __name__ == "__main__":
    main()