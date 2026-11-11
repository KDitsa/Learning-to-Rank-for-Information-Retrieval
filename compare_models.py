import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "features" / "dataset.csv"

MODELS = {
    "Linear SVM": {
        "model": BASE_DIR / "model" / "Pairwise_Linear_Ranking_SVM" / "ranking_svm_model.joblib",
        "metadata": BASE_DIR / "model" / "Pairwise_Linear_Ranking_SVM" / "model_metadata.json",
    },
    "XGBoost Rank:Pairwise": {
        "model": BASE_DIR / "model" / "XGBoost_Rank_Pairwise" / "xgboost_rank_pairwise_model.joblib",
        "metadata": BASE_DIR / "model" / "XGBoost_Rank_Pairwise" / "model_metadata.json",
    },
    "LightGBM LambdaMART": {
        "model": BASE_DIR / "model" / "lambdamart" / "lambdamart_model.joblib",
        "metadata": BASE_DIR / "model" / "lambdamart" / "model_metadata.json",
    },
}

OUTPUT_DIR = BASE_DIR / "model" / "model_comparison"

TEST_SIZE = 0.20
RANDOM_STATE = 42
EXPECTED_FEATURES = 110


def load_dataset():
    df = pd.read_csv(DATASET_PATH)

    rename = {}
    if "qid" not in df.columns and "query_id" in df.columns:
        rename["query_id"] = "qid"
    if "docid" not in df.columns and "doc_id" in df.columns:
        rename["doc_id"] = "docid"

    df = df.rename(columns=rename)

    required = {"qid", "docid", "relevance"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    df["qid"] = df["qid"].astype(str)
    df["relevance"] = pd.to_numeric(df["relevance"]).astype(int)

    excluded = {
        "qid", "query_id", "docid", "doc_id",
        "relevance", "label"
    }

    features = [
        c for c in df.columns
        if c not in excluded
    ]

    if len(features) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, found {len(features)}"
        )

    df[features] = df[features].apply(
        pd.to_numeric, errors="coerce"
    )
    df[features] = df[features].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    if df.duplicated(["qid", "docid"]).any():
        raise ValueError("Duplicate query-document pairs found.")

    return df, features


def get_test_data(df):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(df, groups=df["qid"])
    )

    train_qids = set(df.iloc[train_idx]["qid"])
    test_df = df.iloc[test_idx].copy()

    assert train_qids.isdisjoint(set(test_df["qid"]))

    return test_df


def get_feature_columns(model, metadata_path, dataset_features):
    metadata = {}

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    features = metadata.get("feature_columns")

    if not features:
        features = getattr(model, "feature_names_in_", None)

    if not features and hasattr(model, "booster_"):
        features = model.booster_.feature_name()

    if not features and hasattr(model, "get_booster"):
        features = model.get_booster().feature_names

    if not features:
        features = dataset_features

    features = list(features)

    missing = set(features) - set(dataset_features)
    if missing:
        raise ValueError(
            f"Model expects missing features: {missing}"
        )

    return features


def predict(model, X):
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    else:
        scores = model.predict(X)

    scores = np.asarray(scores).reshape(-1)

    if len(scores) != len(X) or not np.isfinite(scores).all():
        raise ValueError("Invalid model prediction scores.")

    return scores


def dcg(relevances, k):
    rel = np.asarray(relevances, dtype=float)[:k]

    if len(rel) == 0:
        return 0.0

    gains = (2 ** rel) - 1
    discounts = np.log2(np.arange(2, len(rel) + 2))

    return float(np.sum(gains / discounts))


def ndcg_at_k(relevances, k):
    ideal = sorted(relevances, reverse=True)

    ideal_dcg = dcg(ideal, k)

    if ideal_dcg == 0:
        return 0.0

    return dcg(relevances, k) / ideal_dcg


def average_precision(relevances):
    relevant = np.asarray(relevances) > 0
    total_relevant = relevant.sum()

    if total_relevant == 0:
        return 0.0

    precision = np.cumsum(relevant) / np.arange(
        1, len(relevant) + 1
    )

    return float(np.sum(precision * relevant) / total_relevant)


def reciprocal_rank(relevances):
    positions = np.where(np.asarray(relevances) > 0)[0]

    if len(positions) == 0:
        return 0.0

    return 1.0 / (positions[0] + 1)


def pairwise_accuracy(relevances, scores):
    rel = np.asarray(relevances)
    scores = np.asarray(scores)

    i, j = np.triu_indices(len(rel), k=1)

    rel_diff = rel[i] - rel[j]
    valid = rel_diff != 0

    if not valid.any():
        return np.nan

    score_diff = scores[i] - scores[j]

    correct = (
        np.sign(score_diff[valid])
        == np.sign(rel_diff[valid])
    )

    ties = score_diff[valid] == 0

    return float(
        (correct.sum() + 0.5 * ties.sum()) / valid.sum()
    )


def evaluate_model(test_df, scores, model_name):
    df = test_df[
        ["qid", "docid", "relevance"]
    ].copy()

    df["score"] = scores

    per_query_results = []
    ranked_results = []

    for qid, group in df.groupby("qid", sort=False):
        ranked = group.sort_values(
            ["score", "docid"],
            ascending=[False, True],
            kind="mergesort"
        )

        relevances = ranked["relevance"].to_numpy()
        original_relevances = group["relevance"].to_numpy()

        total_relevant = np.sum(original_relevances > 0)

        row = {
            "qid": qid,
            "MAP": average_precision(relevances),
            "MRR": reciprocal_rank(relevances),
            "nDCG@5": ndcg_at_k(relevances, 5),
            "nDCG@10": ndcg_at_k(relevances, 10),
            "pairwise_accuracy": pairwise_accuracy(
                original_relevances,
                group["score"].to_numpy()
            )
        }

        for k in [5, 10]:
            top_k = relevances[:k]
            hits = np.sum(top_k > 0)

            row[f"P@{k}"] = hits / k

            row[f"Recall@{k}"] = (
                hits / total_relevant
                if total_relevant > 0 else 0.0
            )

        per_query_results.append(row)

        ranked = ranked.copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        ranked["model"] = model_name

        ranked_results.append(ranked)

    return (
        pd.DataFrame(per_query_results),
        pd.concat(ranked_results, ignore_index=True)
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading Cranfield dataset...")

    df, dataset_features = load_dataset()
    test_df = get_test_data(df)

    print(f"Total queries: {df['qid'].nunique()}")
    print(f"Test queries: {test_df['qid'].nunique()}")
    print(f"Test query-document pairs: {len(test_df)}")
    print(f"Features: {len(dataset_features)}")

    summary = []
    all_per_query = []
    all_ranked = []

    for model_name, paths in MODELS.items():

        print(f"\nEvaluating {model_name}...")

        model_path = paths["model"]
        metadata_path = paths["metadata"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        model = joblib.load(model_path)
        if model.__class__.__name__ == "XGBRanker":
            model.set_params(device="cpu")

        features = get_feature_columns(
            model,
            metadata_path,
            dataset_features
        )

        X_test = test_df[features].to_numpy(dtype=np.float32)
        scores = predict(model, X_test)
        per_query, ranked = evaluate_model(test_df, scores, model_name)

        all_per_query.append(per_query.assign(model=model_name))
        all_ranked.append(ranked)

        metrics = [
            "MAP",
            "MRR",
            "P@5",
            "P@10",
            "Recall@5",
            "Recall@10",
            "nDCG@5",
            "nDCG@10",
            "pairwise_accuracy"
        ]

        result = {"Model": model_name}

        for metric in metrics:
            result[metric] = per_query[metric].mean()

        summary.append(result)

        print(
            f"MAP: {result['MAP']:.4f} | "
            f"MRR: {result['MRR']:.4f} | "
            f"nDCG@10: {result['nDCG@10']:.4f}"
        )

    comparison = pd.DataFrame(summary)
    comparison = comparison.sort_values(
        "nDCG@10",
        ascending=False
    ).reset_index(drop=True)

    comparison.insert(
        0,
        "Rank",
        np.arange(1, len(comparison) + 1)
    )

    print("\n" + "=" * 90)
    print("FINAL MODEL COMPARISON")
    print("=" * 90)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    best = comparison.iloc[0]

    print("\n" + "=" * 90)
    print("BEST MODEL")
    print("=" * 90)

    print(f"Model: {best['Model']}")
    print(f"MAP: {best['MAP']:.4f}")
    print(f"MRR: {best['MRR']:.4f}")
    print(f"Precision@10: {best['P@10']:.4f}")
    print(f"nDCG@10: {best['nDCG@10']:.4f}")

    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    pd.concat(
        all_per_query,
        ignore_index=True
    ).to_csv(
        OUTPUT_DIR / "per_query_metrics.csv",
        index=False
    )

    pd.concat(
        all_ranked,
        ignore_index=True
    ).to_csv(
        OUTPUT_DIR / "ranked_test_documents.csv",
        index=False
    )

    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
