
"""
XGBoost Rank:Pairwise with Optuna Bayesian Optimization

- Query-level 80/20 train/test split
- GroupKFold cross-validation on training queries
- Optuna TPE hyperparameter optimization
- Optimization objective: mean validation NDCG@10
- Evaluation: Precision, Recall, F1, NDCG, Pairwise Accuracy
"""

from pathlib import Path
import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from xgboost import XGBRanker

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "features" / "dataset.csv"
MODEL_DIR = ROOT / "model" / "XGBoost_Rank_Pairwise"

QID = "qid"
DOCID = "docid"
REL = "relevance"

EXPECTED_FEATURES = 110
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

SELECTION_K = 10
REPORT_KS = (5, 10)

N_TRIALS = 50
OPTUNA_SEED = 42
N_STARTUP_TRIALS = 10


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

    feature_columns = [
        c for c in df.columns if c not in {QID, DOCID, REL}
    ]

    if len(feature_columns) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, found "
            f"{len(feature_columns)}: {feature_columns}"
        )

    df[feature_columns] = df[feature_columns].apply(
        pd.to_numeric, errors="raise"
    )
    df[REL] = pd.to_numeric(df[REL], errors="raise")

    if df[feature_columns].isnull().any().any():
        raise ValueError("Feature columns contain missing values.")

    if not np.isfinite(df[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("Feature columns contain infinite values.")

    if df[REL].isnull().any():
        raise ValueError("Relevance column contains missing values.")

    if not np.equal(
        df[REL].to_numpy(),
        np.floor(df[REL].to_numpy())
    ).all():
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


def prepare_ranking_data(df, feature_columns):
    #XGBoost ranking requires documents belonging to the same query to be contiguous. Group sizes must match this order.
    sorted_df = df.sort_values(QID, kind="stable").reset_index(drop=True)

    group_sizes = (
        sorted_df.groupby(QID, sort=False)
        .size()
        .to_numpy(dtype=np.uint32)
    )

    X = sorted_df[feature_columns].to_numpy(dtype=np.float32)
    y = sorted_df[REL].to_numpy(dtype=np.float32)

    return sorted_df, X, y, group_sizes


def resolve_device():
    try:
        probe = XGBRanker(
            objective="rank:pairwise", n_estimators=1,
            tree_method="hist", device="cuda", verbosity=0
        )
        probe.fit(
            np.random.rand(8, 2).astype(np.float32),
            np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32),
            group=[4, 4],
        )
        return "cuda"
    except Exception:
        return "cpu"

DEVICE = resolve_device()
print(f"Using device: {DEVICE}")


def create_model(params):
    return XGBRanker(
        objective="rank:pairwise",
        eval_metric=f"ndcg@{SELECTION_K}",
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        gamma=params["gamma"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        tree_method="hist",
        device=DEVICE,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )

    
def fit_model(params, train_df, feature_columns):
    _, X, y, group_sizes = prepare_ranking_data(
        train_df, feature_columns
    )

    model = create_model(params)

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        model.fit(X, y, group=group_sizes, verbose=False)

    return model


def score_documents(model, df, feature_columns):
    X = df[feature_columns].to_numpy(dtype=np.float32)
    return model.predict(X)


def per_query_metrics(df, scores, k):
    """
    Per-query P@k, R@k, F1@k and NDCG@k.
    Relevant = relevance > 0.
    Queries with no relevant documents are skipped.
    """
    tmp = pd.DataFrame({
        QID: df[QID].to_numpy(),
        REL: df[REL].to_numpy(),
        "score": np.asarray(scores, dtype=float)
    })

    rows = []

    for qid, group in tmp.groupby(QID, sort=False):
        labels = group.sort_values(
            "score", ascending=False, kind="stable"
        )[REL].to_numpy()

        n_relevant = int((labels > 0).sum())
        if n_relevant == 0:
            continue

        kk = min(k, len(labels))
        top = labels[:kk]
        hits = int((top > 0).sum())

        precision = hits / kk
        recall = hits / n_relevant
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )

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
            "ndcg": ndcg
        })

    return pd.DataFrame(rows)


def summarize(per_query_df):
    if per_query_df.empty:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "ndcg": float("nan"),
            "n_queries": 0
        }

    return {
        "precision": per_query_df["precision"].mean(),
        "recall": per_query_df["recall"].mean(),
        "f1": per_query_df["f1"].mean(),
        "ndcg": per_query_df["ndcg"].mean(),
        "n_queries": len(per_query_df)
    }


def pairwise_accuracy(df, scores):
    """Unequal-relevance pairs ordered correctly; ties count as 0.5."""
    df = df.reset_index(drop=True)
    y_all = df[REL].to_numpy()
    scores = np.asarray(scores, dtype=float)

    correct = 0.0
    total = 0

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


def tune_hyperparameters(train_df, feature_columns):
    n_queries = train_df[QID].nunique()
    n_splits = min(N_SPLITS, n_queries)

    if n_splits < 2:
        raise ValueError("Need at least 2 training queries for CV.")

    gkf = GroupKFold(n_splits=n_splits)
    cv_rows = []

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 600, step=50
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 2, 8
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 20.0, log=True
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.6, 1.0
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "gamma": trial.suggest_float(
                "gamma", 0.0, 5.0
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-8, 10.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-3, 100.0, log=True
            )
        }

        fold_ndcg = []

        print(f"\nTrial {trial.number + 1}/{N_TRIALS}")
        print(f"Parameters: {params}")

        for fold, (fit_idx, val_idx) in enumerate(
            gkf.split(train_df, groups=train_df[QID]),
            start=1
        ):
            fit_df = train_df.iloc[fit_idx].reset_index(drop=True)
            val_df = train_df.iloc[val_idx].reset_index(drop=True)

            model = fit_model(params, fit_df, feature_columns)
            scores = score_documents(model, val_df, feature_columns)

            metrics_df = per_query_metrics(
                val_df, scores, SELECTION_K
            )
            metrics = summarize(metrics_df)
            pair_acc = pairwise_accuracy(val_df, scores)

            fold_ndcg.append(metrics["ndcg"])

            cv_rows.append({
                "trial": trial.number,
                "fold": fold,
                **params,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "ndcg": metrics["ndcg"],
                "pairwise_accuracy": pair_acc
            })

            print(
                f"  Fold {fold}: NDCG@{SELECTION_K}="
                f"{metrics['ndcg']:.4f}, "
                f"P={metrics['precision']:.4f}, "
                f"R={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"PairAcc={pair_acc:.4f}"
            )

        mean_ndcg = float(np.mean(fold_ndcg))
        trial.set_user_attr(
            "fold_ndcg", [float(value) for value in fold_ndcg]
        )

        print(
            f"Trial {trial.number + 1}: "
            f"mean NDCG@{SELECTION_K}={mean_ndcg:.4f}"
        )

        return mean_ndcg

    sampler = optuna.samplers.TPESampler(
        seed=OPTUNA_SEED,
        n_startup_trials=N_STARTUP_TRIALS
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="xgboost_rank_pairwise"
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    best_params = study.best_params

    print("\n========== OPTUNA RESULTS ==========")
    print(f"Best trial: {study.best_trial.number}")
    print(
        f"Best mean validation NDCG@{SELECTION_K}: "
        f"{study.best_value:.4f}"
    )
    print("\nBest hyperparameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    cv_df = pd.DataFrame(cv_rows)

    metric_columns = [
        "precision", "recall", "f1", "ndcg", "pairwise_accuracy"
    ]
    parameter_columns = list(best_params.keys())

    metric_summary = (
        cv_df.groupby("trial")[metric_columns]
        .agg(["mean", "std"])
    )

    parameter_summary = (
        cv_df.groupby("trial")[parameter_columns]
        .first()
    )

    metric_summary.columns = [
        "_".join(str(x) for x in col if str(x))
        if isinstance(col, tuple)
        else str(col)
        for col in metric_summary.columns
    ]

    cv_summary = parameter_summary.join(metric_summary)

    return best_params, cv_df, cv_summary, study


def build_feature_importance(model, feature_columns):
    booster = model.get_booster()

    gain_scores = booster.get_score(importance_type="gain")
    weight_scores = booster.get_score(importance_type="weight")
    cover_scores = booster.get_score(importance_type="cover")

    rows = []

    for i, feature in enumerate(feature_columns):
        key = f"f{i}"
        rows.append({
            "feature": feature,
            "gain": gain_scores.get(key, 0.0),
            "weight": weight_scores.get(key, 0.0),
            "cover": cover_scores.get(key, 0.0)
        })

    table = pd.DataFrame(rows)
    total_gain = table["gain"].sum()

    table["importance_percentage"] = (
        100 * table["gain"] / total_gain
        if total_gain > 0 else 0.0
    )

    table = table.sort_values(
        "gain", ascending=False
    ).reset_index(drop=True)

    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def main():
    parser = argparse.ArgumentParser(description="XGBoost Rank:Pairwise with Optuna optimization.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df, feature_columns = load_dataset(args.data)

    # Query-level train/test split
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(df, groups=df[QID])
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    assert set(train_df[QID]).isdisjoint(set(test_df[QID]))

    print(
        f"\nTraining queries: {train_df[QID].nunique()} "
        f"rows: {len(train_df)}"
    )
    print(
        f"Test queries: {test_df[QID].nunique()} "
        f"rows: {len(test_df)}"
    )

    # Optuna hyperparameter optimization
    best_params, cv_df, cv_summary, study = tune_hyperparameters(train_df, feature_columns)

    # Save cross-validation results
    cv_df.to_csv(
        MODEL_DIR / "cross_validation_results.csv",
        index=False
    )
    cv_summary.to_csv(
        MODEL_DIR / "cross_validation_summary.csv"
    )

    # Save Optuna trial history
    study.trials_dataframe().to_csv(
        MODEL_DIR / "optuna_trials.csv",
        index=False
    )

    # Retrain using all training queries
    final_model = fit_model(
        best_params, train_df, feature_columns
    )

    # Evaluate on untouched test queries
    test_scores = score_documents(
        final_model, test_df, feature_columns
    )

    report_rows = []

    for k in REPORT_KS:
        pq = per_query_metrics(test_df, test_scores, k)

        report_rows.append({
            "method": "XGBoost Rank:Pairwise",
            "k": k,
            **summarize(pq)
        })

    report_df = pd.DataFrame(report_rows)
    test_pairwise = pairwise_accuracy(test_df, test_scores)

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Best validation NDCG@{SELECTION_K}: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    print(best_params)
    print("\nTest metrics:")
    print(report_df.round(4).to_string(index=False))
    print(f"\nTest pairwise accuracy: {test_pairwise:.4f}")

    # Save test metrics
    report_df.to_csv(
        MODEL_DIR / "test_metrics.csv",
        index=False
    )

    # Save feature importance
    importance = build_feature_importance(
        final_model, feature_columns
    )
    importance.to_csv(
        MODEL_DIR / "feature_importance.csv",
        index=False
    )

    # Save trained model
    joblib.dump(
        final_model,
        MODEL_DIR / "xgboost_rank_pairwise_model.joblib"
    )

    # Save model metadata
    metadata = {
        "model_type": "XGBoost Rank:Pairwise",
        "objective": "rank:pairwise",
        "optimizer": "Optuna TPE Bayesian Optimization",
        "feature_columns": feature_columns,
        "best_hyperparameters": best_params,
        "n_trials": N_TRIALS,
        "best_validation_ndcg": float(study.best_value),
        "cv_folds": N_SPLITS,
        "selection_metric": f"NDCG@{SELECTION_K}",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "test_pairwise_accuracy": float(test_pairwise),
        "expected_features": EXPECTED_FEATURES
    }

    with open(
        MODEL_DIR / "model_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=4)

    print("\nTop 10 features:")
    print(
        importance[
            ["rank", "feature", "gain", "importance_percentage"]
        ].head(10).round(4).to_string(index=False)
    )

    print(f"\nAll outputs saved in: {MODEL_DIR}")

if __name__ == "__main__":
    main()
