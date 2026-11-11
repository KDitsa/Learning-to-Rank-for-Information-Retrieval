from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessing import preprocess_text

from feature_extraction import (
    FEATURE_NAMES,
    FeatureExtractor,
)

ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = (ROOT / "data" / "preprocessed")
DOCUMENTS_PATH = (PROCESSED_DIR / "documents_processed.csv")

MODEL_DIR = (ROOT / "model" / "lambdamart")
MODEL_PATH = (MODEL_DIR / "lambdamart_model.joblib")
METADATA_PATH = (MODEL_DIR / "model_metadata.json")

OUTPUT_PATH = (ROOT / "lambdamart_query_results.txt")

DEFAULT_TOP_K = 10


def load_model():
    print("Loading trained LambdaMART model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    # Use the feature order established by training.
    feature_columns = list(FEATURE_NAMES)

    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as source:
            metadata = json.load(source)

        if "feature_columns" in metadata:
            saved_columns = metadata["feature_columns"]

            if saved_columns != feature_columns:
                raise ValueError(
                    "Feature order mismatch between "
                    "training metadata and feature extractor."
                )

        print("Training metadata loaded.")
        if "best_parameters" in metadata:
            print("Selected parameters:", metadata["best_parameters"])

    # Check the number of features expected by the model.
    expected_features = getattr(
        model,
        "n_features_in_",
        None,
    )

    if (expected_features is not None and expected_features != len(feature_columns)):
        raise ValueError(
            f"Model expects {expected_features} features, "
            f"but feature extractor provides "
            f"{len(feature_columns)}."
        )

    print("LambdaMART model loaded successfully.")
    print("Features:", len(feature_columns))

    return model, feature_columns


def load_documents():
    print("\nLoading Cranfield documents...")

    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Processed documents not found: {DOCUMENTS_PATH}"
        )

    extractor = FeatureExtractor.from_processed(PROCESSED_DIR)
    print("Documents loaded:", extractor.num_documents)
    return extractor


def process_query(query):
    if not query or not query.strip():
        return []

    return preprocess_text(query)


def rank_documents(
    query_tokens,
    model,
    extractor,
    feature_columns,
):
    if not query_tokens:
        return pd.DataFrame(
            columns=["document_id", "score"]
        )

    feature_rows = []
    document_ids = []

    for doc_id in extractor.document_ids:
        features = extractor.extract(query_tokens, doc_id,)

        # Preserve the exact training feature order.
        feature_rows.append([
            features[name]
            for name in feature_columns
        ])

        document_ids.append(doc_id)

    X = pd.DataFrame(
        feature_rows,
        columns=feature_columns,
        dtype=np.float64,
    )

    if X.shape != (len(document_ids), len(feature_columns),):
        raise ValueError(
            "Unexpected feature matrix dimensions."
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Feature matrix contains NaN or infinite values."
        )

    scores = np.asarray(model.predict(X)).reshape(-1)
    if len(scores) != len(document_ids):
        raise ValueError(
            "Prediction count does not match document count."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "LambdaMART produced non-finite scores."
        )

    results = pd.DataFrame({
        "document_id": document_ids,
        "score": scores,
    })

    results = results.sort_values(
        by=["score", "document_id"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    results.insert(
        0,
        "rank",
        np.arange(1, len(results) + 1),
    )

    return results


def save_results(
    results,
    output_path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True,)
    with output_path.open("w", encoding="utf-8") as output:
        for doc_id in results["document_id"]:
            output.write(
                f"{int(doc_id)}\n"
            )

    print("\nResults saved to:", output_path)


def display_results(
    results,
    query,
    query_tokens,
):
    print("\n" + "=" * 75)
    print("LAMBDAMART RANKED SEARCH RESULTS")
    print("=" * 75)

    print("\nOriginal query:", query)
    print("Processed query:", ", ".join(query_tokens))

    if results.empty:
        print("\nNo results found.")
        return

    print("\nRanked documents:\n")
    print(results.head(20).to_string(index=False))


def search_single_query(
    query,
    model,
    extractor,
    feature_columns,
    top_k,
    output_path,
):
    query_tokens = process_query(query)

    if not query_tokens:
        print("\nNo searchable terms remain after preprocessing.")
        return

    results = rank_documents(query_tokens, model, extractor, feature_columns)

    # Select top-K documents.
    results = results.head(top_k)
    display_results(results, query, query_tokens)
    save_results(results, output_path)


def search_query_file(
    query_file,
    model,
    extractor,
    feature_columns,
    top_k,
    output_path,
):
    query_file = Path(query_file)
    if not query_file.exists():
        raise FileNotFoundError(
            f"Query file not found: {query_file}"
        )

    with query_file.open("r", encoding="utf-8",) as source:
        queries = [
            line.strip()
            for line in source
            if line.strip()
        ]

    if not queries:
        raise ValueError(
            "Query file contains no queries."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True,)

    with output_path.open("w",encoding="utf-8",) as output:
        for query_number, query in enumerate(queries, start=1):
            print(f"\nProcessing query {query_number}/{len(queries)}")
            query_tokens = process_query(query)
            output.write(f"Query {query_number}: {query}\n")

            if not query_tokens:
                print("Skipping: no searchable terms.")
                output.write("No searchable terms.\n\n")
                continue

            results = rank_documents(query_tokens, model, extractor, feature_columns)
            results = results.head(top_k)

            for doc_id in results["document_id"]:
                output.write(f"{int(doc_id)}\n")
            output.write("\n")

            print(f"Completed query {query_number}.")

    print("\nBatch results saved to:", output_path,)


def main():

    parser = argparse.ArgumentParser(description=("Cranfield Information Retrieval using the tuned LambdaMART model"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--query", type=str, help="Free-text search query")
    group.add_argument("--query-file", type=Path, help="File containing one query per line")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of top documents to retrieve")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output file for ranked document IDs")
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive.")

    # Load the trained model.
    model, feature_columns = load_model()

    # Load the processed document collection.
    extractor = load_documents()

    print("\nSearch system initialized successfully.")

    if args.query_file is None:
        query = args.query
        if query is None:
            print("\nEnter a free-text query.")
            print("Type 'exit' or 'quit' to close.")

            while True:
                try:
                    query = input("\nQuery> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting search system.")
                    break

                if query.lower() in {"exit", "quit", "q"}:
                    print("\nExiting search system.")
                    break

                if not query:
                    continue

                try:
                    search_single_query(query, model, extractor, feature_columns, args.top_k, args.output)
                except Exception as error:
                    print(f"\nSearch error: {error}")

        else:
            if not query.strip():
                parser.error("Query cannot be empty.")
            search_single_query(query, model, extractor, feature_columns, args.top_k, args.output)

    else:
        try:
            search_query_file(args.query_file, model, extractor, feature_columns, args.top_k, args.output)
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()