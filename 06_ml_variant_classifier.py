"""Train a supervised spelling-variant classifier and export interpretable rules."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

try:
    from importlib.util import spec_from_file_location, module_from_spec

    _feature_path = Path(__file__).with_name("05_feature_extractor.py")
    _feature_spec = spec_from_file_location("spelling_feature_extractor", _feature_path)
    if _feature_spec is None or _feature_spec.loader is None:
        raise ImportError(f"Cannot load feature module at {_feature_path}.")
    _feature_module = module_from_spec(_feature_spec)
    _feature_spec.loader.exec_module(_feature_module)
    FEATURE_COLUMNS: tuple[str, ...] = _feature_module.FEATURE_COLUMNS
except ImportError:
    raise


@dataclass
class TrainingResult:
    """Fitted estimators, cross-validation scores, and feature ordering."""

    classifier: RandomForestClassifier
    rule_tree: DecisionTreeClassifier
    cv_scores: np.ndarray
    feature_columns: tuple[str, ...]
    threshold: float = 0.5


def tune_threshold(
    classifier: RandomForestClassifier,
    features: pd.DataFrame,
    labels: Sequence[int],
    candidate_thresholds: Iterable[float] | None = None,
) -> float:
    """Choose the threshold maximizing validation F1 while prioritizing precision."""
    from sklearn.metrics import precision_recall_fscore_support

    target = np.asarray(labels, dtype=int)
    class_index = list(classifier.classes_).index(1)
    probabilities = classifier.predict_proba(features[list(FEATURE_COLUMNS)])[:, class_index]
    thresholds = list(candidate_thresholds or np.arange(0.10, 0.91, 0.05))
    best_threshold = 0.5
    best_key = (-1.0, -1.0)
    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            target, predicted, average="binary", zero_division=0
        )
        key = (precision, f1)
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def export_uncertain_pairs(
    classifier: RandomForestClassifier,
    candidate_features: pd.DataFrame,
    output_path: str | Path,
    limit: int = 100,
    similarity_column: str = "character_ngram_tfidf_cosine",
) -> pd.DataFrame:
    """Export pairs nearest the decision boundary, prioritizing similar pairs."""
    if limit < 1:
        raise ValueError("limit must be positive.")
    if not {"word1", "word2"}.issubset(candidate_features.columns):
        raise ValueError("Candidate features must include word1 and word2.")
    missing = set(FEATURE_COLUMNS) - set(candidate_features.columns)
    if missing:
        raise ValueError(f"Candidate features are missing columns: {sorted(missing)}")
    probabilities = classifier.predict_proba(candidate_features[list(FEATURE_COLUMNS)])
    positive_index = list(classifier.classes_).index(1)
    output = candidate_features[["word1", "word2"]].copy()
    output["variant_probability"] = probabilities[:, positive_index]
    output["similarity"] = (
        candidate_features[similarity_column]
        if similarity_column in candidate_features
        else 0.0
    )
    output["uncertainty"] = np.abs(output["variant_probability"] - 0.5)
    output = output.sort_values(
        ["uncertainty", "similarity"], ascending=[True, False]
    ).head(limit)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def train_classifier(
    features: pd.DataFrame,
    labels: Sequence[int],
    n_estimators: int = 200,
    n_splits: int = 5,
    random_state: int = 42,
    tree_max_depth: int = 5,
) -> TrainingResult:
    """Cross-validate and fit a random forest plus a shallow rules tree."""
    if n_estimators < 1 or n_splits < 2 or tree_max_depth < 1:
        raise ValueError("Estimator count, folds, and tree depth must be positive.")
    missing = set(FEATURE_COLUMNS) - set(features.columns)
    if missing:
        raise ValueError(f"Training features are missing columns: {sorted(missing)}")
    target = np.asarray(labels, dtype=int)
    if len(target) != len(features):
        raise ValueError("Feature and label counts must match.")
    classes, class_counts = np.unique(target, return_counts=True)
    if len(classes) != 2 or not set(classes).issubset({0, 1}):
        raise ValueError("Training labels must contain both binary classes 0 and 1.")
    folds = min(n_splits, int(class_counts.min()))
    if folds < 2:
        raise ValueError("Each class needs at least two labeled examples for cross-validation.")

    x_values = features[list(FEATURE_COLUMNS)]
    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    cross_validator = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=random_state
    )
    scores = cross_val_score(
        estimator, x_values, target, cv=cross_validator, scoring="accuracy"
    )
    estimator.fit(x_values, target)
    rules = DecisionTreeClassifier(
        max_depth=tree_max_depth, random_state=random_state, class_weight="balanced"
    ).fit(x_values, target)
    return TrainingResult(estimator, rules, scores, FEATURE_COLUMNS)


def extract_decision_rules(result: TrainingResult) -> str:
    """Return readable decision-tree conditions in feature-name order."""
    return export_text(
        result.rule_tree,
        feature_names=list(result.feature_columns),
        decimals=3,
    )


def save_training_artifacts(
    result: TrainingResult,
    model_path: str | Path,
    threshold: float,
) -> None:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as stream:
        pickle.dump(
            {
                "classifier": result.classifier,
                "rule_tree": result.rule_tree,
                "feature_columns": result.feature_columns,
                "threshold": float(threshold),
                "cv_scores": result.cv_scores.tolist(),
            },
            stream,
        )


def load_training_artifacts(model_path: str | Path) -> dict[str, object]:
    with Path(model_path).open("rb") as stream:
        return pickle.load(stream)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--n-splits", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tree-max-depth", type=int, default=3)
    args = parser.parse_args()

    from sklearn.datasets import make_classification

    values, labels = make_classification(
        n_samples=40, n_features=len(FEATURE_COLUMNS), n_informative=6,
        n_redundant=0, random_state=args.random_state
    )
    sample = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    result = train_classifier(
        sample, labels, args.n_estimators, args.n_splits,
        args.random_state, args.tree_max_depth
    )
    print(f"cross_validation_accuracy={result.cv_scores.mean():.3f}")
    print(extract_decision_rules(result))
    return 0


if __name__ == "__main__":
    assert callable(train_classifier) and callable(export_uncertain_pairs)
    raise SystemExit(main())
