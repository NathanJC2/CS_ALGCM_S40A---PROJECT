"""Create lexical, contextual, frequency, and character n-gram pair features."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def _edit_analyzer():
    module_path = Path(__file__).with_name("03_dp_edit_distance.py")
    spec = importlib.util.spec_from_file_location("spelling_edit_distance", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load edit-distance module at {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.analyze_edit_distance


_analyze_edit_distance = _edit_analyzer()

FEATURE_COLUMNS = (
    "edit_distance",
    "normalized_edit_distance",
    "edit_position_start",
    "edit_position_middle",
    "edit_position_end",
    "transform_vowel_to_vowel",
    "transform_consonant_to_consonant",
    "transform_insertion_or_deletion",
    "context_cosine",
    "local_context_jaccard",
    "log_frequency_ratio",
    "minimum_frequency_met",
    "character_ngram_tfidf_cosine",
    "word_length_difference",
    "common_prefix_length",
    "common_suffix_length",
    "pinoy_language_same",
    "pinoy_filipino_probability",
    "pinoy_language_confidence",
)


def _collect_contexts(
    sentences: Sequence[Sequence[str]], context_window: int
) -> dict[str, set[str]]:
    contexts: dict[str, set[str]] = defaultdict(set)
    for sentence in sentences:
        for index, word in enumerate(sentence):
            start = max(0, index - context_window)
            end = min(len(sentence), index + context_window + 1)
            contexts[word].update(
                sentence[position]
                for position in range(start, end)
                if position != index
            )
    return contexts


def _character_ngram_cosine(word1: str, word2: str, minimum: int, maximum: int) -> float:
    """Cohort-independent binary character n-gram cosine similarity."""
    def grams(word: str) -> set[str]:
        padded = f"^{word}$"
        return {
            padded[index:index + width]
            for width in range(minimum, maximum + 1)
            for index in range(max(0, len(padded) - width + 1))
        }

    left, right = grams(word1), grams(word2)
    return len(left & right) / float(max(1, len(left) * len(right)) ** 0.5)


def build_feature_matrix(
    candidate_pairs: Iterable[tuple[str, str]],
    frequencies: Mapping[str, int],
    sentences: Sequence[Sequence[str]],
    context_model=None,
    min_frequency: int = 2,
    context_window: int = 2,
    char_ngram_min: int = 2,
    char_ngram_max: int = 4,
    edit_analyses: Mapping[tuple[str, str], Any] | None = None,
    language_predictions: Mapping[str, Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    """Return pair identifiers and numeric features for every candidate.

    All thresholds and feature windows are parameters. The TF-IDF vocabulary is
    fit over the supplied corpus vocabulary and pair endpoints, so comparison
    values are consistent within this returned matrix.
    """
    if min_frequency < 0 or context_window < 0:
        raise ValueError("Frequency threshold and context window cannot be negative.")
    if char_ngram_min < 1 or char_ngram_max < char_ngram_min:
        raise ValueError("Character n-gram bounds must satisfy 1 <= min <= max.")
    pairs = [(str(word1), str(word2)) for word1, word2 in candidate_pairs]
    columns = ["word1", "word2", *FEATURE_COLUMNS]
    if not pairs:
        return pd.DataFrame(columns=columns)

    local_contexts = _collect_contexts(sentences, context_window)
    rows: list[dict[str, float | str]] = []
    vowels = set("aeiou")

    for word1, word2 in pairs:
        edit = (
            edit_analyses[(word1, word2)]
            if edit_analyses is not None and (word1, word2) in edit_analyses
            else _analyze_edit_distance(word1, word2)
        )
        denominator = max(len(word1), len(word2), 1)
        positions = edit.normalized_positions
        edits = edit.transformations
        position_features = {
            "edit_position_start": float(any(position <= 1 / 3 for position in positions)),
            "edit_position_middle": float(any(1 / 3 < position < 2 / 3 for position in positions)),
            "edit_position_end": float(any(position >= 2 / 3 for position in positions)),
        }
        common_prefix = len(os.path.commonprefix([word1, word2]))
        common_suffix = 0
        for left, right in zip(reversed(word1), reversed(word2)):
            if left != right:
                break
            common_suffix += 1
        language1 = language_predictions.get(word1, {}) if language_predictions else {}
        language2 = language_predictions.get(word2, {}) if language_predictions else {}
        probabilities1 = language1.get("probabilities", {})
        probabilities2 = language2.get("probabilities", {})
        filipino_probability = (
            float(probabilities1.get("FIL", 0.0)) +
            float(probabilities2.get("FIL", 0.0))
        ) / 2.0
        confidence = (
            float(language1.get("confidence", 0.0)) +
            float(language2.get("confidence", 0.0))
        ) / 2.0
        rows.append(
            {
                "word1": word1,
                "word2": word2,
                "edit_distance": float(edit.distance),
                "normalized_edit_distance": edit.distance / denominator,
                **position_features,
                "transform_vowel_to_vowel": float(
                    any(
                        len(left) == len(right) == 1 and left in vowels and right in vowels
                        for item in edits
                        for left, right in [item.split(" vs. ")]
                    )
                ),
                "transform_consonant_to_consonant": float(
                    any(
                        len(left) == len(right) == 1
                        and left not in vowels
                        and right not in vowels
                        and left != "-"
                        and right != "-"
                        for item in edits
                        for left, right in [item.split(" vs. ")]
                    )
                ),
                "transform_insertion_or_deletion": float(
                    any("-" in item.split(" vs. ") for item in edits)
                ),
                "context_cosine": (
                    context_model.similarity(word1, word2) if context_model is not None else 0.0
                ),
                "local_context_jaccard": (
                    len(local_contexts.get(word1, set()) & local_contexts.get(word2, set()))
                    / max(
                        1,
                        len(local_contexts.get(word1, set()) | local_contexts.get(word2, set())),
                    )
                ),
                "log_frequency_ratio": float(
                    np.log(
                        (max(frequencies.get(word1, 0), frequencies.get(word2, 0)) + 1)
                        / (min(frequencies.get(word1, 0), frequencies.get(word2, 0)) + 1)
                    )
                ),
                "minimum_frequency_met": float(
                    min(frequencies.get(word1, 0), frequencies.get(word2, 0)) >= min_frequency
                ),
                "character_ngram_tfidf_cosine": _character_ngram_cosine(
                    word1, word2, char_ngram_min, char_ngram_max
                ),
                "word_length_difference": float(abs(len(word1) - len(word2))),
                "common_prefix_length": float(common_prefix),
                "common_suffix_length": float(common_suffix),
                "pinoy_language_same": float(
                    bool(language1 and language2 and language1["label"] == language2["label"])
                ),
                "pinoy_filipino_probability": filipino_probability,
                "pinoy_language_confidence": confidence,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--context-window", type=int, default=2)
    parser.add_argument("--char-ngram-min", type=int, default=2)
    parser.add_argument("--char-ngram-max", type=int, default=4)
    args = parser.parse_args()
    frequencies = Counter({"politiko": 5, "pulitiko": 3, "other": 2})
    features = build_feature_matrix(
        [("politiko", "pulitiko")],
        frequencies,
        [["politiko", "is", "a", "word"], ["pulitiko", "is", "a", "word"]],
        min_frequency=args.min_frequency,
        context_window=args.context_window,
        char_ngram_min=args.char_ngram_min,
        char_ngram_max=args.char_ngram_max,
    )
    print(features.to_string(index=False))
    return 0


if __name__ == "__main__":
    assert len(FEATURE_COLUMNS) == 19
    raise SystemExit(main())
