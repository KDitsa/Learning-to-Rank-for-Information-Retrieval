import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from optuna.samplers import TPESampler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "features" / "dataset.csv"
OUTPUT_DIR = BASE_DIR / "model" / "lambdamart"

EXPECTED_FEATURES = 110
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_SPLITS = 5
N_TRIALS = 50

NDCG_K = 10
PRECISION_K = 10

LABEL_GAIN = [0, 1, 3, 7, 15]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_device():
    rng = np.random.default_rng(0)
    X = rng.random((100, 5))
    y = rng.integers(0, 2, 100)
    try:
        lgb.LGBMRanker(
            objective="lambdarank", n_estimators=2,
            min_child_samples=2, device_type="gpu", verbosity=-1
        ).fit(X, y, group=[50, 50])
        return "gpu"
    except Exception:
        return "cpu"

DEVICE_TYPE = resolve_device()

def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    # Support both common naming conventions.
    rename_map = {}

    if "qid" not in df.columns and "query_id" in df.columns:
        rename_map["query_id"] = "qid"

    if "docid" not in df.columns and "doc_id" in df.columns:
        rename_map["doc_id"] = "docid"

    df = df.rename(columns=rename_map)

    required = {"qid", "docid", "relevance"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    excluded = {
        "qid",
        "docid",
        "relevance",
        "query_id",
        "doc_id",
        "label"
    }

    feature_columns = [
        col for col in df.columns
        if col not in excluded
    ]

    if len(feature_columns) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, "
            f"but found {len(feature_columns)}."
        )

    df[feature_columns] = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    df[feature_columns] = df[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    df["relevance"] = pd.to_numeric(
        df["relevance"],
        errors="raise"
    )

    if df["relevance"].isna().any():
        raise ValueError("Relevance column contains missing values.")

    if (df["relevance"] < 0).any():
        raise ValueError(
            "LambdaMART requires non-negative relevance labels. "
            "Check your qrels relevance mapping."
        )

    if not np.allclose(
        df["relevance"],
        np.round(df["relevance"])
    ):
        raise ValueError(
            "Relevance labels must be integers for this setup."
        )

    df["relevance"] = df["relevance"].astype(int)

    max_label = int(df["relevance"].max())

    if max_label >= len(LABEL_GAIN):
        raise ValueError(
            f"Maximum relevance label is {max_label}, "
            f"but LABEL_GAIN supports labels 0 to "
            f"{len(LABEL_GAIN) - 1}. Update LABEL_GAIN."
        )

    df["qid"] = df["qid"].astype(str)

    if df.duplicated(["qid", "docid"]).any():
        raise ValueError(
            "Duplicate query-document pairs found in dataset."
        )

    print(f"Dataset shape: {df.shape}")
    print(f"Queries: {df['qid'].nunique()}")
    print(f"Documents: {df['docid'].nunique()}")
    print(f"Features: {len(feature_columns)}")
    print(f"Relevance labels: {sorted(df['relevance'].unique())}")

    return df, feature_columns


def prepare_ranking_data(df, feature_columns):
    # LightGBM requires rows belonging to the same query to be contiguous. The group array contains the number of documents belonging to each query.
    df = df.sort_values(
        "qid",
        kind="mergesort"
    ).reset_index(drop=True)

    X = df[feature_columns]
    y = df["relevance"].to_numpy(dtype=np.int32)

    group_sizes = (
        df.groupby("qid", sort=False)
        .size()
        .to_numpy()
        .tolist()
    )

    assert sum(group_sizes) == len(df)

    return df, X, y, group_sizes


def calculate_dcg(relevances, k=10):
    relevances = np.asarray(relevances, dtype=float)[:k]

    if len(relevances) == 0:
        return 0.0

    gains = (2 ** relevances) - 1
    discounts = np.log2(np.arange(2, len(relevances) + 2))

    return float(np.sum(gains / discounts))


def calculate_ndcg(relevances, k=10):
    actual_dcg = calculate_dcg(relevances, k)

    ideal_relevances = sorted(
        relevances,
        reverse=True
    )

    ideal_dcg = calculate_dcg(ideal_relevances, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def calculate_pairwise_accuracy(relevances, scores):
    relevances = np.asarray(relevances)
    scores = np.asarray(scores)

    n = len(relevances)

    if n < 2:
        return np.nan

    i, j = np.triu_indices(n, k=1)

    relevance_diff = relevances[i] - relevances[j]

    valid = relevance_diff != 0

    if not np.any(valid):
        return np.nan

    score_diff = scores[i] - scores[j]

    correct = (
        np.sign(score_diff[valid])
        == np.sign(relevance_diff[valid])
    )

    ties = score_diff[valid] == 0

    accuracy = (
        np.sum(correct) + 0.5 * np.sum(ties)
    ) / len(correct)

    return float(accuracy)


def evaluate_queries(df, scores):
    """
    Calculates ranking metrics independently for each query.
    """

    evaluation_df = df[
        ["qid", "docid", "relevance"]
    ].copy()

    evaluation_df["score"] = np.asarray(scores)

    results = []

    for qid, group in evaluation_df.groupby(
        "qid",
        sort=False
    ):
        ranked = group.sort_values(
            "score",
            ascending=False
        )

        relevances = ranked["relevance"].to_numpy()
        total_relevant = int(np.sum(relevances > 0))

        top_k = relevances[:PRECISION_K]

        precision = (
            np.sum(top_k > 0) / PRECISION_K
        )

        recall = (
            np.sum(top_k > 0) / total_relevant
            if total_relevant > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        ndcg = calculate_ndcg(
            relevances,
            NDCG_K
        )

        pairwise_acc = calculate_pairwise_accuracy(
            group["relevance"].to_numpy(),
            group["score"].to_numpy()
        )

        results.append({
            "qid": qid,
            "precision@10": precision,
            "recall@10": recall,
            "f1@10": f1,
            "ndcg@10": ndcg,
            "pairwise_accuracy": pairwise_acc
        })

    return pd.DataFrame(results)


def summarize_metrics(per_query_df):
    return {
        metric: float(per_query_df[metric].mean())
        for metric in [
            "precision@10",
            "recall@10",
            "f1@10",
            "ndcg@10",
            "pairwise_accuracy"
        ]
    }


def create_model(params):
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[10],
        lambdarank_truncation_level=10,
        label_gain=LABEL_GAIN,
        device_type=DEVICE_TYPE,
        verbosity=-1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        importance_type="gain",
        **params
    )

    return model


def fit_model(model, df, feature_columns):
    sorted_df, X, y, group_sizes = prepare_ranking_data(
        df,
        feature_columns
    )

    model.fit(
        X,
        y,
        group=group_sizes
    )

    return model


def split_dataset(df):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            groups=df["qid"]
        )
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_queries = set(train_df["qid"])
    test_queries = set(test_df["qid"])

    assert train_queries.isdisjoint(test_queries)

    print("\nTrain/test split")
    print(f"Training queries: {len(train_queries)}")
    print(f"Testing queries: {len(test_queries)}")
    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows: {len(test_df)}")

    return train_df, test_df


cv_rows = []


def objective(trial, train_df, feature_columns):
    max_depth = trial.suggest_int(
        "max_depth",
        3,
        8
    )

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            600,
            step=50
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.2,
            log=True
        ),

        "max_depth": max_depth,

        "num_leaves": trial.suggest_int(
            "num_leaves",
            7,
            min(63, 2 ** max_depth)
        ),

        "min_child_samples": trial.suggest_int(
            "min_child_samples",
            5,
            30
        ),

        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            0.001,
            10.0,
            log=True
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),

        "subsample_freq": 1,

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-8,
            10.0,
            log=True
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-3,
            100.0,
            log=True
        ),

        "min_split_gain": trial.suggest_float(
            "min_split_gain",
            0.0,
            1.0
        )
    }

    group_kfold = GroupKFold(
        n_splits=N_SPLITS
    )

    fold_ndcg_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        group_kfold.split(
            train_df,
            groups=train_df["qid"]
        ),
        start=1
    ):
        fold_train = train_df.iloc[train_idx].copy()
        fold_val = train_df.iloc[val_idx].copy()

        model = create_model(params)

        fit_model(
            model,
            fold_train,
            feature_columns
        )

        val_sorted, X_val, _, _ = prepare_ranking_data(
            fold_val,
            feature_columns
        )

        predictions = model.predict(X_val)

        fold_metrics = evaluate_queries(
            val_sorted,
            predictions
        )

        metrics = summarize_metrics(fold_metrics)

        fold_ndcg_scores.append(
            metrics["ndcg@10"]
        )

        cv_rows.append({
            "trial": trial.number,
            "fold": fold,
            **params,
            **metrics
        })

    mean_ndcg = float(np.mean(fold_ndcg_scores))

    trial.set_user_attr(
        "mean_ndcg@10",
        mean_ndcg
    )

    return mean_ndcg


def optimize_hyperparameters(train_df, feature_columns):
    sampler = TPESampler(
        seed=RANDOM_STATE,
        n_startup_trials=10
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="cranfield_lambdamart"
    )

    print("\nStarting Optuna optimization...")
    print(f"Trials: {N_TRIALS}")
    print(f"Cross-validation folds: {N_SPLITS}")
    print(f"Device: {DEVICE_TYPE}")

    study.optimize(
        lambda trial: objective(
            trial,
            train_df,
            feature_columns
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    print("\nBest CV NDCG@10:")
    print(study.best_value)

    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    return study


def save_cv_results(study):
    cv_df = pd.DataFrame(cv_rows)

    cv_results_path = (
        OUTPUT_DIR / "cross_validation_results.csv"
    )

    cv_df.to_csv(
        cv_results_path,
        index=False
    )

    metric_columns = [
        "precision@10",
        "recall@10",
        "f1@10",
        "ndcg@10",
        "pairwise_accuracy"
    ]

    cv_summary = (
        cv_df.groupby("trial")[metric_columns]
        .agg(["mean", "std"])
    )

    cv_summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in cv_summary.columns
    ]

    cv_summary = cv_summary.reset_index()

    cv_summary_path = (
        OUTPUT_DIR / "cross_validation_summary.csv"
    )

    cv_summary.to_csv(
        cv_summary_path,
        index=False
    )

    trials_df = study.trials_dataframe()

    trials_df.to_csv(
        OUTPUT_DIR / "optuna_trials.csv",
        index=False
    )

    print("\nSaved cross-validation results.")


def train_final_model(
    train_df,
    test_df,
    feature_columns,
    study
):
    best_params = study.best_params

    print("\nTraining final LambdaMART model...")

    final_model = create_model(best_params)

    fit_model(
        final_model,
        train_df,
        feature_columns
    )

    test_sorted, X_test, _, _ = prepare_ranking_data(
        test_df,
        feature_columns
    )

    test_predictions = final_model.predict(X_test)

    per_query_test = evaluate_queries(
        test_sorted,
        test_predictions
    )

    test_metrics = summarize_metrics(
        per_query_test
    )

    print("\nFinal test metrics")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.6f}")

    # Save aggregate test metrics.
    test_metrics_df = pd.DataFrame([
        {
            "metric": metric,
            "value": value
        }
        for metric, value in test_metrics.items()
    ])

    test_metrics_df.to_csv(
        OUTPUT_DIR / "test_metrics.csv",
        index=False
    )

    # Save per-query test metrics.
    per_query_test.to_csv(
        OUTPUT_DIR / "per_query_test_metrics.csv",
        index=False
    )

    # Save feature importance.
    booster = final_model.booster_

    gain_importance = booster.feature_importance(
        importance_type="gain"
    )

    split_importance = booster.feature_importance(
        importance_type="split"
    )

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "gain_importance": gain_importance,
        "split_importance": split_importance
    })

    total_gain = importance_df["gain_importance"].sum()

    if total_gain > 0:
        importance_df["gain_percentage"] = (
            importance_df["gain_importance"]
            / total_gain
            * 100
        )
    else:
        importance_df["gain_percentage"] = 0.0

    importance_df = importance_df.sort_values(
        "gain_importance",
        ascending=False
    )

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False
    )

    # Save trained model.
    joblib.dump(
        final_model,
        OUTPUT_DIR / "lambdamart_model.joblib"
    )

    # Save metadata.
    metadata = {
        "model": "LightGBM LambdaMART",
        "objective": "lambdarank",
        "optimizer": "Optuna TPE",
        "device_type": DEVICE_TYPE,
        "expected_features": EXPECTED_FEATURES,
        "feature_columns": feature_columns,
        "best_parameters": best_params,
        "best_cv_ndcg_at_10": study.best_value,
        "n_trials": N_TRIALS,
        "cv_folds": N_SPLITS,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "selection_metric": "mean CV NDCG@10",
        "test_metrics": test_metrics
    }

    with open(
        OUTPUT_DIR / "model_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4
        )

    print("\nSaved final model and results.")
    print(f"Output directory: {OUTPUT_DIR}")

    return final_model


def main():
    print("LightGBM version:", lgb.__version__)
    print("Configured device:", DEVICE_TYPE)

    df, feature_columns = load_dataset()

    train_df, test_df = split_dataset(df)

    study = optimize_hyperparameters(
        train_df,
        feature_columns
    )

    save_cv_results(study)

    train_final_model(
        train_df,
        test_df,
        feature_columns,
        study
    )

    print("\nLambdaMART pipeline completed successfully.")


if __name__ == "__main__":
    main()