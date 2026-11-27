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
    labels_csv: str | Path | None,
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
    variant_model_path: str | Path = "variant_classifier.pkl",
    test_limit: int | None = None,
    evaluation_xlsx: str | Path | None = None,
    evaluation_sample_size: int = 100,
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
    if test_limit is not None:
        if test_limit < 1:
            raise ValueError("test_limit must be positive")
        sentences = sentences[:test_limit]
        allowed = {word for sentence in sentences for word in sentence}
        frequencies = pd.Series(
            [sum(sentence.count(word) for sentence in sentences) for word in allowed],
            index=sorted(allowed),
        ).to_dict()
    language_predictions = _timed(
        "PinoyBot language identification",
        lambda: preprocess.tag_vocabulary(frequencies),
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

    labeled_pairs: list[tuple[str, str]] = []
    labels: list[int] = []
    if labels_csv is not None:
        labels_frame = pd.read_csv(labels_csv)
        required = {"word1", "word2", "is_variant"}
        if not required.issubset(labels_frame.columns):
            raise ValueError(f"Label CSV must contain columns {sorted(required)}.")
        labels = pd.to_numeric(labels_frame["is_variant"], errors="raise").astype(int).tolist()
        if not set(labels).issubset({0, 1}):
            raise ValueError("is_variant labels must be binary values 0 or 1.")
        labeled_pairs = list(zip(
            labels_frame["word1"].astype(str),
            labels_frame["word2"].astype(str),
        ))
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
            language_predictions=language_predictions,
        ),
    )
    training_features = combined_features.iloc[:len(labeled_pairs)].reset_index(drop=True)
    model_path = Path(variant_model_path)
    if labeled_pairs:
        training = _timed(
            "06 classifier training",
            lambda: classifier.train_classifier(
                training_features,
                labels,
                n_estimators=n_estimators,
                n_splits=n_splits,
                random_state=random_state,
            ),
        )
        selected_threshold = _timed(
            "06 threshold tuning",
            lambda: classifier.tune_threshold(
                training.classifier, training_features, labels,
            ),
        )
        classifier.save_training_artifacts(
            training, model_path, selected_threshold,
        )
    elif model_path.is_file():
        artifact = _timed(
            "06 load classifier",
            lambda: classifier.load_training_artifacts(model_path),
        )
        training = None
        selected_threshold = float(artifact["threshold"])
        trained_classifier = artifact["classifier"]
        trained_feature_columns = tuple(artifact["feature_columns"])
    else:
        raise FileNotFoundError(
            "No verified word-pair labels or existing variant_classifier.pkl were provided. "
            "Create labels with word1,word2,is_variant and run training explicitly."
        )

    candidate_features = combined_features.iloc[len(labeled_pairs):].reset_index(drop=True)
    def predict_and_resolve():
        if candidate_features.empty:
            return []
        active_classifier = (
            training.classifier if training is not None else trained_classifier
        )
        feature_columns = (
            list(features.FEATURE_COLUMNS)
            if training is not None
            else list(trained_feature_columns)
        )
        class_index = list(active_classifier.classes_).index(1)
        probabilities = active_classifier.predict_proba(
            candidate_features[feature_columns]
        )[:, class_index]
        scored = [
            (row.word1, row.word2, float(probability))
            for row, probability in zip(candidate_features.itertuples(index=False), probabilities)
            if probability >= selected_threshold
        ]
        return resolver.resolve_variant_pairs(
            scored,
            minimum_confidence=selected_threshold,
            cohesion_threshold=cohesion_threshold,
        )

    resolved = _timed("07 greedy resolution", predict_and_resolve)
    exported_count = _timed(
        "08 CSV export",
        lambda: exporter.export_variants(resolved, dict(frequencies), output_csv),
    )
    evaluated_pairs = 0
    if evaluation_xlsx is not None:
        sampler = _load_stage("09_sampler_for_evaluation.py", "pipeline_sampler")
        evaluated_pairs = _timed(
            "09 evaluation sample",
            lambda: sampler.create_evaluation_sample(
                output_csv,
                evaluation_xlsx,
                sample_size=evaluation_sample_size,
                random_state=random_state,
            ),
        )
    summary = {
        "vocabulary_size": len(frequencies),
        "candidate_pairs": len(candidates),
        "labeled_pairs": len(labeled_pairs),
        "resolved_pairs": len(resolved),
        "exported_pairs": exported_count,
        "cross_validation_accuracy_mean": (
            float(training.cv_scores.mean()) if training is not None else None
        ),
        "prediction_threshold": float(selected_threshold),
        "evaluation_sample_pairs": evaluated_pairs,
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
    parser.add_argument("--variant-model", default="variant_classifier.pkl")
    parser.add_argument("--test-limit", type=int, help="Limit preprocessing to this many sentences")
    parser.add_argument("--evaluation-xlsx", help="Optional Evaluation.xlsx output")
    parser.add_argument("--evaluation-sample-size", type=int, default=100)
    args = parser.parse_args()
    if args.demo:
        _run_synthetic_demo()
        return 0
    if not args.corpus or not args.output_csv:
        parser.error("--corpus and --output-csv are required unless --demo is used.")
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
        variant_model_path=args.variant_model,
        test_limit=args.test_limit,
        evaluation_xlsx=args.evaluation_xlsx,
        evaluation_sample_size=args.evaluation_sample_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
