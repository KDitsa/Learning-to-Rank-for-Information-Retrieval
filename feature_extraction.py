from __future__ import annotations

import csv
import json
import math

from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Optional

FIELDS = ("body", "author", "title", "bibliography")
STREAMS = ("body", "author", "title", "bibliography", "whole_document")

DEFAULT_PROCESSED_DIR = (Path(__file__).resolve().parent / "data"/ "preprocessed")

BASE_FEATURE_NAMES = (
    "covered_query_term_number",
    "covered_query_term_ratio",
    "stream_length",
    "idf",

    "tf_sum",
    "tf_min",
    "tf_max",
    "tf_mean",
    "tf_variance",

    "normalized_tf_sum",
    "normalized_tf_min",
    "normalized_tf_max",
    "normalized_tf_mean",
    "normalized_tf_variance",

    "tfidf_sum",
    "tfidf_min",
    "tfidf_max",
    "tfidf_mean",
    "tfidf_variance",

    "boolean_model",
    "vector_space_model",
    "bm25",
)

FEATURE_NAMES = tuple(
    f"{stream}_{feature}"
    for stream in STREAMS
    for feature in BASE_FEATURE_NAMES
)

assert len(BASE_FEATURE_NAMES) == 22
assert len(FEATURE_NAMES) == 110


def _tokens(values: Iterable[str]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError("Pass token strings as a sequence, not a text string")

    result = list(values)
    if any(not isinstance(t, str) or not t for t in result):
        raise ValueError("Tokens must be non-empty strings")
    return result


def _decode(value: str) -> list[str]:
    result = json.loads(value)
    if not isinstance(result, list):
        raise ValueError("A token column must contain a JSON list")

    return _tokens(result)


def _stats(values: list[float]) -> tuple[float, ...]:
    if not values:
        return (0.0,) * 5

    total = math.fsum(values)
    minimum = min(values)
    maximum = max(values)
    mean = total / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)

    return total, minimum, maximum, mean, variance


def _log_tf(count: int) -> float:
    return 1.0 + math.log10(count) if count > 0 else 0.0


def load_query_tokens(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> dict[int, list[str]]:
    queries = {}

    path = Path(processed_dir) / "queries_processed.csv"

    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            qid = int(row["query_id"])

            if qid in queries:
                raise ValueError(f"Duplicate query ID: {qid}")

            queries[qid] = _decode(row["query_stemmed_tokens"])

    return queries


class FeatureExtractor:
    def __init__(
        self,
        documents: Mapping[int, Mapping[str, Iterable[str]]],
        *,
        k1: float = 1.2,
        b: float = 0.75,
        k3: Optional[float] = None,
        boolean_mode: str = "or",
    ) -> None:

        if not documents:
            raise ValueError("Document collection must not be empty")

        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be finite and positive")

        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be in [0, 1]")

        if k3 is not None and (not math.isfinite(k3) or k3 <= 0):
            raise ValueError("k3 must be None or finite and positive")

        if boolean_mode not in ("or", "and"):
            raise ValueError("boolean_mode must be 'or' or 'and'")

        self.k1 = k1
        self.b = b
        self.k3 = k3
        self.boolean_mode = boolean_mode

        self.document_ids = tuple(sorted(documents))
        self.num_documents = len(documents)

        # Store token counts separately for every stream.
        self._counts = {
            stream: {}
            for stream in STREAMS
        }

        self._lengths = {
            stream: {}
            for stream in STREAMS
        }

        self._idf = {}
        self._norms = {}
        self._average_length = {}

        # Build field-specific streams.
        for doc_id, fields in documents.items():

            missing = set(FIELDS) - set(fields)

            if missing:
                raise ValueError(
                    f"Document {doc_id} missing fields: {missing}"
                )

            normalized_fields = {
                field: _tokens(fields[field])
                for field in FIELDS
            }

            # Whole document is the concatenation of all fields.
            normalized_fields["whole_document"] = [
                token
                for field in FIELDS
                for token in normalized_fields[field]
            ]

            for stream in STREAMS:
                counts = Counter(
                    normalized_fields[stream]
                )

                self._counts[stream][doc_id] = counts
                self._lengths[stream][doc_id] = sum(
                    counts.values()
                )

        # Fit corpus statistics independently for each stream.
        for stream in STREAMS:

            counts_by_doc = self._counts[stream]
            lengths = self._lengths[stream]

            average_length = (
                math.fsum(lengths.values()) / self.num_documents
            )

            self._average_length[stream] = average_length

            # Document frequency.
            df = Counter()

            for counts in counts_by_doc.values():
                df.update(counts.keys())

            # Unsmoothed IDF: log10(N / df).
            self._idf[stream] = {
                term: math.log10(self.num_documents / frequency)
                for term, frequency in df.items()
            }

            # Precompute document TF-IDF vector norms.
            self._norms[stream] = {}

            for doc_id, counts in counts_by_doc.items():

                norm = math.sqrt(
                    math.fsum(
                        (
                            _log_tf(tf)
                            * self._idf[stream][term]
                        ) ** 2
                        for term, tf in counts.items()
                    )
                )

                self._norms[stream][doc_id] = norm


    @classmethod
    def from_processed(
        cls,
        processed_dir: Path = DEFAULT_PROCESSED_DIR,
        **kwargs,
    ) -> "FeatureExtractor":
        documents = {}
        path = Path(processed_dir) / "documents_processed.csv"

        with path.open(newline="",encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                doc_id = int(row["doc_id"])

                if doc_id in documents:
                    raise ValueError(f"Duplicate document ID: {doc_id}")

                documents[doc_id] = {
                    field: _decode(
                        row[f"{field}_stemmed_tokens"]
                    )
                    for field in FIELDS
                }

        return cls(documents, **kwargs)


    def _extract_stream(
        self,
        query: Counter,
        doc_id: int,
        stream: str,
    ) -> tuple[float, ...]:

        counts = self._counts[stream][doc_id]
        length = self._lengths[stream][doc_id]

        terms = sorted(query)

        if not terms:
            return (0.0,) * 22

        # Query-term document frequencies and IDF.
        idf_map = self._idf[stream]

        idfs = [
            idf_map.get(term, 0.0)
            for term in terms
        ]

        # Raw TF for every unique query term.
        tf = [
            float(counts.get(term, 0))
            for term in terms
        ]

        covered = sum(value > 0 for value in tf)
        coverage_ratio = covered / len(terms)

        # Stream-length normalized TF.
        normalized_tf = [
            value / length if length else 0.0
            for value in tf
        ]

        # Log-TF multiplied by IDF.
        tfidf = [
            _log_tf(int(counts.get(term, 0))) * idf
            for term, idf in zip(terms, idfs)
        ]

        # IDF feature: sum of query-term IDF.
        idf_sum = math.fsum(idfs)

        # TF, normalized TF and TF-IDF statistics.
        tf_stats = _stats(tf)
        normalized_stats = _stats(normalized_tf)
        tfidf_stats = _stats(tfidf)

        # Boolean OR / AND matching.
        if self.boolean_mode == "or":
            boolean_score = float(covered > 0)
        else:
            boolean_score = float(covered == len(terms))

        # Vector space model: cosine similarity.
        query_weights = [
            _log_tf(query[term]) * idf
            for term, idf in zip(terms, idfs)
        ]

        query_norm = math.sqrt(
            math.fsum(weight ** 2 for weight in query_weights)
        )

        document_norm = self._norms[stream][doc_id]
        denominator = query_norm * document_norm

        cosine = (
            math.fsum(
                q_weight * d_weight
                for q_weight, d_weight
                in zip(query_weights, tfidf)
            ) / denominator
            if denominator
            else 0.0
        )

        # Protect against floating-point roundoff.
        cosine = min(1.0, max(0.0, cosine))

        # BM25.
        bm25_terms = []

        average_length = self._average_length[stream]

        if length and average_length:

            length_normalization = (
                self.k1
                * (
                    1 - self.b
                    + self.b * length / average_length
                )
            )

            for term in terms:

                frequency = counts.get(term, 0)
                idf = idf_map.get(term, 0.0)

                if frequency == 0 or idf == 0:
                    continue

                denominator = length_normalization + frequency

                query_factor = (
                    (
                        (self.k3 + 1) * query[term]
                        / (self.k3 + query[term])
                    )
                    if self.k3 is not None
                    else 1.0
                )

                score = (
                    idf
                    * (self.k1 + 1)
                    * frequency
                    / denominator
                    * query_factor
                )

                bm25_terms.append(score)

        bm25_score = math.fsum(bm25_terms)

        # Return exactly 22 features in fixed order.
        return (
            float(covered),
            coverage_ratio,
            float(length),
            idf_sum,

            *tf_stats,
            *normalized_stats,
            *tfidf_stats,

            boolean_score,
            cosine,
            bm25_score,
        )


    def extract(
        self,
        query_tokens: Iterable[str],
        doc_id: int,
    ) -> dict[str, float]:
        """Return a named 110-feature dictionary."""

        if doc_id not in self.document_ids:
            raise KeyError(f"Unknown document ID: {doc_id}")

        query = Counter(_tokens(query_tokens))

        features = {}

        for stream in STREAMS:

            values = self._extract_stream(
                query=query,
                doc_id=doc_id,
                stream=stream,
            )

            names = [
                f"{stream}_{name}"
                for name in BASE_FEATURE_NAMES
            ]

            features.update(zip(names, values))

        # Ensure fixed feature order and finite values.
        result = {
            name: float(features[name])
            for name in FEATURE_NAMES
        }

        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError("Non-finite feature value detected")

        return result
