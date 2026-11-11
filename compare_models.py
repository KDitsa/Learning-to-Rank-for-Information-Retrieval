import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
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

# BM25 baseline: the whole-document BM25 feature used on its own as the ranking score.
BASELINE_NAME = "BM25 (baseline)"
BASELINE_FEATURE = "whole_document_bm25"

OUTPUT_DIR = BASE_DIR / "model" / "model_comparison"

TEST_SIZE = 0.20
RANDOM_STATE = 42
EXPECTED_FEATURES = 110
ALPHA = 0.05

METRICS = [
    "MAP",
    "MRR",
    "P@5",
    "P@10",
    "Recall@5",
    "Recall@10",
    "nDCG@5",
    "nDCG@10",
    "pairwise_accuracy",
]

# Metrics used for the paired significance tests.
TEST_METRICS = ["nDCG@10", "MAP"]


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

    if BASELINE_FEATURE not in features:
        raise ValueError(
            f"Baseline feature '{BASELINE_FEATURE}' not found in dataset."
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

    def empty(x):
        return x is None or len(x) == 0

    features = metadata.get("feature_columns")

    if empty(features):
        features = getattr(model, "feature_names_in_", None)

    if empty(features) and hasattr(model, "booster_"):
        features = model.booster_.feature_name()

    if empty(features) and hasattr(model, "get_booster"):
        features = model.get_booster().feature_names

    if empty(features):
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
        score_q = group["score"].to_numpy()
        original_relevances = group["relevance"].to_numpy()

        # Pessimistic tie-breaking: higher score first; among equal scores,
        # LOWER relevance first, so ties never help the ranker.
        order = np.lexsort((original_relevances, -score_q))
        ranked = group.iloc[order]

        relevances = ranked["relevance"].to_numpy()

        total_relevant = np.sum(original_relevances > 0)

        row = {
            "qid": qid,
            "MAP": average_precision(relevances),
            "MRR": reciprocal_rank(relevances),
            "nDCG@5": ndcg_at_k(relevances, 5),
            "nDCG@10": ndcg_at_k(relevances, 10),
            "pairwise_accuracy": pairwise_accuracy(
                original_relevances,
                score_q
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


def paired_wilcoxon_tests(per_query_all, model_names):
    """
    Paired Wilcoxon signed-rank tests on per-query metrics.
    Each ranker is compared with the BM25 baseline, and
    the learned rankers are compared with each other.
    """
    pairs = [(m, BASELINE_NAME) for m in model_names]
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            pairs.append((model_names[i], model_names[j]))

    rows = []

    for metric in TEST_METRICS:
        table = per_query_all.pivot(
            index="qid", columns="model", values=metric
        )

        for a, b in pairs:
            x = table[a].to_numpy()
            y = table[b].to_numpy()
            diff = x - y

            if np.allclose(diff, 0):
                p_value = np.nan
            else:
                p_value = float(wilcoxon(x, y).pvalue)

            rows.append({
                "metric": metric,
                "model_A": a,
                "model_B": b,
                "mean_A": x.mean(),
                "mean_B": y.mean(),
                "mean_diff (A-B)": diff.mean(),
                "queries_A_better": int((diff > 0).sum()),
                "queries_B_better": int((diff < 0).sum()),
                "queries_tied": int((diff == 0).sum()),
                "p_value": p_value,
                f"significant@{ALPHA}": (
                    bool(p_value < ALPHA)
                    if not np.isnan(p_value) else False
                ),
            })

    return pd.DataFrame(rows)


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

    def record(name, per_query, ranked):
        all_per_query.append(per_query.assign(model=name))
        all_ranked.append(ranked)

        result = {"Model": name}
        for metric in METRICS:
            result[metric] = per_query[metric].mean()
        summary.append(result)

        print(
            f"MAP: {result['MAP']:.4f} | "
            f"MRR: {result['MRR']:.4f} | "
            f"nDCG@10: {result['nDCG@10']:.4f}"
        )

    # ---- BM25 baseline ----
    print(f"\nEvaluating {BASELINE_NAME} ({BASELINE_FEATURE})...")
    baseline_scores = test_df[BASELINE_FEATURE].to_numpy(dtype=float)
    per_query, ranked = evaluate_model(
        test_df, baseline_scores, BASELINE_NAME
    )
    record(BASELINE_NAME, per_query, ranked)

    # ---- Learned rankers ----
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
        record(model_name, per_query, ranked)

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
    print("FINAL MODEL COMPARISON (including BM25 baseline)")
    print("=" * 90)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    learned = comparison[comparison["Model"] != BASELINE_NAME]
    best = learned.iloc[0]

    print("\n" + "=" * 90)
    print("BEST LEARNED MODEL (by test nDCG@10)")
    print("=" * 90)

    print(f"Model: {best['Model']}")
    print(f"MAP: {best['MAP']:.4f}")
    print(f"MRR: {best['MRR']:.4f}")
    print(f"Precision@10: {best['P@10']:.4f}")
    print(f"nDCG@10: {best['nDCG@10']:.4f}")

    # ---- Paired Wilcoxon signed-rank tests ----
    per_query_all = pd.concat(all_per_query, ignore_index=True)

    tests = paired_wilcoxon_tests(per_query_all, list(MODELS.keys()))

    print("\n" + "=" * 90)
    print(f"PAIRED WILCOXON SIGNED-RANK TESTS "
          f"(per-query, {test_df['qid'].nunique()} test queries)")
    print("=" * 90)

    print(
        tests.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )
    print(
        "\nNote: p-values are not corrected for multiple comparisons."
    )

    # ---- Save everything ----
    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    per_query_all.to_csv(
        OUTPUT_DIR / "per_query_metrics.csv",
        index=False
    )

    tests.to_csv(
        OUTPUT_DIR / "significance_tests.csv",
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