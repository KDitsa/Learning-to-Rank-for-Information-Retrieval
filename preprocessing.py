from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "preprocessed"

DOCUMENTS_INPUT = RAW_DIR / "cran.all.1400"
QUERIES_INPUT = RAW_DIR / "cran.qry"

FIELDS = ("title", "author", "bibliography", "body")
TAG_TO_FIELD = {
    ".T": "title",
    ".A": "author",
    ".B": "bibliography",
    ".W": "body",
}

DOCUMENT_ID_RE = re.compile(r"\.I\s+(\d+)")
THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
RAW_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*['’]?|\d+(?:\.\d+)?")
WORD_RE = re.compile(r"[a-z]+")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

STEMMER = PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM)


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    text = THOUSANDS_COMMA_RE.sub("", text)
    return RAW_TOKEN_RE.findall(text)


def normalize(tokens: list[str]) -> list[str]:
    result = []

    for token in tokens:
        value = unicodedata.normalize("NFKC", token).lower().replace("’", "'")

        if value.endswith("'s") and value[:-2].isalpha():
            value = value[:-2]
        elif value.endswith("'"):
            value = value[:-1]

        value = value.replace("'", "")

        if WORD_RE.fullmatch(value) or NUMBER_RE.fullmatch(value):
            result.append(value)

    return result


try:
    STOPWORDS = {
        STEMMER.stem(token)
        for word in stopwords.words("english")
        for token in normalize(tokenize(word))
        if WORD_RE.fullmatch(token)
    }
except LookupError as exc:
    raise RuntimeError(
        "NLTK stopwords corpus is missing. Run: "
        "python -m nltk.downloader stopwords"
    ) from exc


def preprocess_text(text: str) -> list[str]:
    result = []

    for token in normalize(tokenize(text)):
        processed = STEMMER.stem(token) if WORD_RE.fullmatch(token) else token

        if processed not in STOPWORDS:
            result.append(processed)

    return result


def parse_documents(path: Path) -> list[tuple[int, dict[str, list[str]]]]:
    documents = []
    current_id = None
    current_field = None
    fields = {field: [] for field in FIELDS}
    seen_fields = set()

    def finish_document():
        missing = set(FIELDS) - seen_fields

        if missing:
            raise ValueError(
                f"Document {current_id} missing fields: {sorted(missing)}"
            )

        documents.append((current_id, fields.copy()))

    with Path(path).open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.rstrip("\r\n")
            marker = line.strip()
            match = DOCUMENT_ID_RE.fullmatch(marker)

            if match:
                if current_id is not None:
                    finish_document()

                current_id = int(match.group(1))
                current_field = None
                fields = {field: [] for field in FIELDS}
                seen_fields = set()
                continue

            if current_id is None:
                if marker:
                    raise ValueError(
                        f"Text before first document at {path}:{line_number}"
                    )
                continue

            if marker in TAG_TO_FIELD:
                current_field = TAG_TO_FIELD[marker]
                seen_fields.add(current_field)
                continue

            if marker.startswith("."):
                current_field = None
                continue

            if current_field is not None:
                fields[current_field].extend(preprocess_text(line))

    if current_id is not None:
        finish_document()

    if not documents:
        raise ValueError(f"No documents found in {path}")

    return documents


def parse_queries(path: Path) -> list[tuple[int, list[str]]]:
    queries = []
    current_id = None
    text_parts = []

    def finish_query():
        queries.append(
            (len(queries) + 1, preprocess_text(" ".join(text_parts)))
        )

    with Path(path).open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.rstrip("\r\n")
            marker = line.strip()
            match = DOCUMENT_ID_RE.fullmatch(marker)

            if match:
                if current_id is not None:
                    finish_query()

                current_id = True
                text_parts = []
                continue

            if current_id is None:
                if marker:
                    raise ValueError(
                        f"Text before first query at {path}:{line_number}"
                    )
                continue

            if marker.startswith("."):
                continue

            text_parts.append(line)

    if current_id is not None:
        finish_query()

    if not queries:
        raise ValueError(f"No queries found in {path}")

    return queries
    

def write_documents(documents, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "doc_id",
        "title_stemmed_tokens",
        "author_stemmed_tokens",
        "bibliography_stemmed_tokens",
        "body_stemmed_tokens",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for doc_id, fields in documents:
            writer.writerow({
                "doc_id": doc_id,
                **{
                    f"{field}_stemmed_tokens": json.dumps(fields[field])
                    for field in FIELDS
                },
            })


def write_queries(queries, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["query_id", "query_stemmed_tokens"],
        )
        writer.writeheader()

        for query_id, tokens in queries:
            writer.writerow({
                "query_id": query_id,
                "query_stemmed_tokens": json.dumps(tokens),
            })


def main():
    parser = argparse.ArgumentParser(description="Preprocess Cranfield documents and queries")
    parser.add_argument("--documents", type=Path, default=DOCUMENTS_INPUT)
    parser.add_argument("--queries", type=Path, default=QUERIES_INPUT)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)

    args = parser.parse_args()

    print("Processing Cranfield documents...")
    documents = parse_documents(args.documents)

    print("Processing Cranfield queries...")
    queries = parse_queries(args.queries)

    write_documents(
        documents,
        args.output_dir / "documents_processed.csv",
    )
    write_queries(
        queries,
        args.output_dir / "queries_processed.csv",
    )

    print("\nPreprocessing completed successfully!")
    print(f"Documents processed: {len(documents)}")
    print(f"Queries processed: {len(queries)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
