"""CLI runner connecting preprocessing, blocking, modeling, filtering, and export.

The supervised stage expects a user-supplied CSV containing ``word1``, ``word2``,
and binary ``is_variant`` labels (0/1). Paths are explicit CLI arguments.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd


def _load_stage(numbered_filename: str, module_name: str):
    module_path = Path(__file__).with_name(numbered_filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pipeline stage at {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _timed(stage: str, operation: Callable[[], object]):
    started = time.perf_counter()
    result = operation()
    elapsed = time.perf_counter() - started
    print(f"[{stage}] {elapsed:.3f}s")
    return result


def run_pipeline(
    corpus_paths: Sequence[str | Path],
    labels_csv: str | Path,
    output_csv: str | Path,
    encoding: str = "utf-8",
    max_length_diff: int = 1,
    leaf_size: int = 64,
    max_pairs_per_signature: int = 5000,
    embedding_backend: str = "auto",
    vector_size: int = 50,
    context_window: int = 2,
    embedding_window: int = 5,
    embedding_components: int = 50,
    embedding_epochs: int = 10,
    min_frequency: int = 2,
    char_ngram_min: int = 2,
    char_ngram_max: int = 4,
    n_estimators: int = 200,
    n_splits: int = 5,
    random_state: int = 42,
    prediction_threshold: float = 0.5,
    cohesion_threshold: float = 0.75,
) -> dict[str, object]:
    """Run the full pipeline and return stage counts and validation summary."""
    if not 0.0 <= prediction_threshold <= 1.0:
        raise ValueError("prediction_threshold must be between zero and one.")
    preprocess = _load_stage("01_preprocess_vocab.py", "pipeline_preprocess")
    blocking = _load_stage("02_candidate_blocking_dnc.py", "pipeline_blocking")
    edit_distance = _load_stage("03_dp_edit_distance.py", "pipeline_edit_distance")
    embeddings = _load_stage("04_train_context_embeddings.py", "pipeline_embeddings")
    features = _load_stage("05_feature_extractor.py", "pipeline_features")
    classifier = _load_stage("06_ml_variant_classifier.py", "pipeline_classifier")
    resolver = _load_stage("07_greedy_cluster_resolver.py", "pipeline_resolver")
    exporter = _load_stage("08_export_variants_csv.py", "pipeline_exporter")

    frequencies, sentences = _timed(
        "01 preprocess", lambda: preprocess.read_corpus(corpus_paths, encoding)
    )
    candidates = _timed(
        "02 candidate blocking",
        lambda: blocking.generate_candidates(
            frequencies.keys(),
            max_length_diff=max_length_diff,
            leaf_size=leaf_size,
            max_pairs_per_signature=max_pairs_per_signature,
        ),
    )
    context_model = _timed(
        "04 context model",
        lambda: embeddings.train_context_model(
            sentences,
            backend=embedding_backend,
            vector_size=vector_size,
            window_size=embedding_window,
            components=embedding_components,
            epochs=embedding_epochs,
        ),
    )

    labels_frame = pd.read_csv(labels_csv)
    required = {"word1", "word2", "is_variant"}
    if not required.issubset(labels_frame.columns):
        raise ValueError(
            f"Label CSV must contain columns {sorted(required)}."
        )
    labels = pd.to_numeric(labels_frame["is_variant"], errors="raise").astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError("is_variant labels must be binary values 0 or 1.")
    labeled_pairs = list(zip(labels_frame["word1"].astype(str), labels_frame["word2"].astype(str)))
    all_pairs = labeled_pairs + candidates
    edit_analyses = _timed(
        "03 dynamic programming",
        lambda: {
            pair: edit_distance.analyze_edit_distance(*pair)
            for pair in all_pairs
        },
    )
    combined_features = _timed(
        "05 feature extraction",
        lambda: features.build_feature_matrix(
            all_pairs,
            frequencies,
            sentences,
            context_model=context_model,
            min_frequency=min_frequency,
            context_window=context_window,
            char_ngram_min=char_ngram_min,
            char_ngram_max=char_ngram_max,
            edit_analyses=edit_analyses,
        ),
    )
    training_features = combined_features.iloc[:len(labeled_pairs)].reset_index(drop=True)
    training = _timed(
        "06 classifier training",
        lambda: classifier.train_classifier(
            training_features,
            labels.tolist(),
            n_estimators=n_estimators,
            n_splits=n_splits,
            random_state=random_state,
        ),
    )

    candidate_features = combined_features.iloc[len(labeled_pairs):].reset_index(drop=True)
    def predict_and_resolve():
        if candidate_features.empty:
            return []
        class_index = list(training.classifier.classes_).index(1)
        probabilities = training.classifier.predict_proba(
            candidate_features[list(features.FEATURE_COLUMNS)]
        )[:, class_index]
        scored = [
            (row.word1, row.word2, float(probability))
            for row, probability in zip(candidate_features.itertuples(index=False), probabilities)
            if probability >= prediction_threshold
        ]
        return resolver.resolve_variant_pairs(
            scored,
            minimum_confidence=prediction_threshold,
            cohesion_threshold=cohesion_threshold,
        )

    resolved = _timed("07 greedy resolution", predict_and_resolve)
    exported_count = _timed(
        "08 CSV export",
        lambda: exporter.export_variants(resolved, dict(frequencies), output_csv),
    )
    summary = {
        "vocabulary_size": len(frequencies),
        "candidate_pairs": len(candidates),
        "labeled_pairs": len(labeled_pairs),
        "resolved_pairs": len(resolved),
        "exported_pairs": exported_count,
        "cross_validation_accuracy_mean": float(training.cv_scores.mean()),
        "output_csv": str(output_csv),
        "embedding_backend": context_model.backend,
    }
    print(f"[summary] {summary}")
    return summary


def _run_synthetic_demo() -> None:
    """Run a tiny, temporary end-to-end corpus for an independent smoke test."""
    with tempfile.TemporaryDirectory() as directory:
        corpus_handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", dir=directory, delete=False
        )
        labels_handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".csv", dir=directory, delete=False
        )
        output_handle = tempfile.NamedTemporaryFile(
            suffix=".csv", dir=directory, delete=False
        )
        corpus_path = Path(corpus_handle.name)
        labels_path = Path(labels_handle.name)
        output_path = Path(output_handle.name)
        corpus_handle.write(
            "alpha alfa shared context. beta theta separate setting. " * 3
        )
        corpus_handle.close()
        labels_handle.close()
        output_handle.close()
        pd.DataFrame(
            [
                ("alpha", "alfa", 1),
                ("beta", "theta", 0),
                ("alpha", "beta", 0),
                ("alfa", "theta", 0),
                ("alpha", "alfa", 1),
                ("beta", "theta", 0),
            ],
            columns=["word1", "word2", "is_variant"],
        ).to_csv(labels_path, index=False)
        run_pipeline(
            [corpus_path],
            labels_path,
            output_path,
            embedding_backend="sklearn",
            min_frequency=1,
            n_estimators=10,
            n_splits=2,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", nargs="+", help="Input .txt/.csv files")
    parser.add_argument("--labels-csv", help="CSV with word1, word2, is_variant")
    parser.add_argument("--output-csv", help="Final variants CSV path")
    parser.add_argument(
        "--demo", action="store_true", help="Run the pipeline on a temporary synthetic corpus"
    )
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--max-length-diff", type=int, default=1)
    parser.add_argument("--leaf-size", type=int, default=64)
    parser.add_argument("--max-pairs-per-signature", type=int, default=5000)
    parser.add_argument("--embedding-backend", choices=("auto", "gensim", "sklearn"), default="auto")
    parser.add_argument("--vector-size", type=int, default=50)
    parser.add_argument("--embedding-window", type=int, default=5)
    parser.add_argument("--embedding-components", type=int, default=50)
    parser.add_argument("--embedding-epochs", type=int, default=10)
    parser.add_argument("--context-window", type=int, default=2)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--char-ngram-min", type=int, default=2)
    parser.add_argument("--char-ngram-max", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--cohesion-threshold", type=float, default=0.75)
    args = parser.parse_args()
    if args.demo:
        _run_synthetic_demo()
        return 0
    if not args.corpus or not args.labels_csv or not args.output_csv:
        parser.error("--corpus, --labels-csv, and --output-csv are required unless --demo is used.")
    run_pipeline(
        args.corpus,
        args.labels_csv,
        args.output_csv,
        encoding=args.encoding,
        max_length_diff=args.max_length_diff,
        leaf_size=args.leaf_size,
        max_pairs_per_signature=args.max_pairs_per_signature,
        embedding_backend=args.embedding_backend,
        vector_size=args.vector_size,
        context_window=args.context_window,
        embedding_window=args.embedding_window,
        embedding_components=args.embedding_components,
        embedding_epochs=args.embedding_epochs,
        min_frequency=args.min_frequency,
        char_ngram_min=args.char_ngram_min,
        char_ngram_max=args.char_ngram_max,
        n_estimators=args.n_estimators,
        n_splits=args.n_splits,
        random_state=args.random_state,
        prediction_threshold=args.prediction_threshold,
        cohesion_threshold=args.cohesion_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
