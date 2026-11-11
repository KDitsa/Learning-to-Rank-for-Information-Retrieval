import argparse
import csv
from pathlib import Path

from feature_extraction import FEATURE_NAMES, FeatureExtractor, load_query_tokens


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "preprocessed"

DEFAULT_QRELS = RAW_DIR / "cranqrel"
DEFAULT_OUTPUT = DATA_DIR / "features" / "dataset.csv"


RELEVANCE_MAPPING = {
    -1: 0,
    1: 4,
    2: 3,
    3: 2,
    4: 1,
}


def load_qrels(qrels_path):
    qrels = {}

    with Path(qrels_path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 3:
                raise ValueError(
                    f"Invalid qrels line {line_number}: expected 3 columns, "
                    f"got {len(parts)}"
                )

            try:
                qid = int(parts[0])
                did = int(parts[1])
                original_label = int(parts[2])

            except ValueError as exc:
                raise ValueError(
                    f"Invalid values at qrels line {line_number}: {line}"
                ) from exc

            if original_label not in RELEVANCE_MAPPING:
                raise ValueError(
                    f"Unknown relevance label {original_label} "
                    f"at line {line_number}"
                )

            key = (qid, did)

            if key in qrels:
                raise ValueError(
                    f"Duplicate qrels entry for query {qid}, document {did}"
                )

            qrels[key] = RELEVANCE_MAPPING[original_label]

    return qrels


def build_dataset(
    qrels_path=DEFAULT_QRELS,
    output_path=DEFAULT_OUTPUT,
    processed_dir=PROCESSED_DIR,
):
    print("Loading processed documents...")
    extractor = FeatureExtractor.from_processed(processed_dir)

    print("Loading processed queries...")
    queries = load_query_tokens(processed_dir)

    print("Loading relevance judgments...")
    qrels = load_qrels(qrels_path)

    document_ids = set(extractor.document_ids)
    query_ids = set(queries)

    # Validate qrels IDs against processed files.
    for qid, did in qrels:
        if qid not in query_ids:
            raise ValueError(f"Query ID {qid} not found in processed queries")

        if did not in document_ids:
            raise ValueError(f"Document ID {did} not found in processed documents")

    fieldnames = [
        "query_id",
        "doc_id",
        *FEATURE_NAMES,
        "relevance",
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_counts = {label: 0 for label in range(5)}

    judged_queries = sorted({qid for qid, _ in qrels})
    all_documents = sorted(document_ids)

    print(f"Extracting features for {len(judged_queries)} queries x "
          f"{len(all_documents)} documents...")

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for n, qid in enumerate(judged_queries, start=1):
            query_tokens = queries[qid]

            for did in all_documents:
                relevance = qrels.get((qid, did), 0)      # unjudged = 0
                features = extractor.extract(query_tokens, did)

                writer.writerow({
                    "query_id": qid,
                    "doc_id": did,
                    **features,
                    "relevance": relevance,
                })
                label_counts[relevance] += 1

            if n % 25 == 0:
                print(f"  {n}/{len(judged_queries)} queries done")

    print("\nDataset created successfully!")
    print("Output:", output_path)
    print("Queries:", len(judged_queries))
    print("Documents:", len(all_documents))
    print("Total dataset rows:", sum(label_counts.values()))
    print("\nGain distribution (0 = unjudged):")
    for label, count in label_counts.items():
        print(f"Gain {label}: {count}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build Cranfield Linear SVM dataset")
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS,)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR,)

    args = parser.parse_args()

    build_dataset(qrels_path=args.qrels, output_path=args.output, processed_dir=args.processed_dir,)


if __name__ == "__main__":
    main()
