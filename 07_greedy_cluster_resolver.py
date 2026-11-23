"""Greedily retain high-confidence pairs while preventing transitive drift."""

from __future__ import annotations

import argparse
from typing import Iterable


def resolve_variant_pairs(
    weighted_pairs: Iterable[tuple[str, str, float]],
    minimum_confidence: float = 0.5,
    cohesion_threshold: float = 0.75,
) -> list[tuple[str, str, float]]:
    """Merge components only when every cross-component pair is sufficiently similar.

    Missing direct evidence is treated as insufficient cohesion. This complete-
    linkage guard prevents a chain of individually plausible edges from joining
    words whose mutual similarity is low or unknown.
    """
    if not 0.0 <= minimum_confidence <= 1.0 or not 0.0 <= cohesion_threshold <= 1.0:
        raise ValueError("Confidence thresholds must be between zero and one.")
    best_scores: dict[tuple[str, str], float] = {}
    for word1, word2, score in weighted_pairs:
        if not 0.0 <= float(score) <= 1.0:
            raise ValueError("Pair confidence scores must be between zero and one.")
        if word1 == word2:
            continue
        pair = tuple(sorted((word1, word2)))
        best_scores[pair] = max(best_scores.get(pair, float("-inf")), float(score))

    clusters: dict[str, set[str]] = {}
    membership: dict[str, str] = {}
    retained: set[tuple[str, str, float]] = set()
    for (word1, word2), score in sorted(
        best_scores.items(), key=lambda item: (-item[1], item[0])
    ):
        if score < minimum_confidence:
            continue
        root1 = membership.get(word1, word1)
        root2 = membership.get(word2, word2)
        cluster1 = clusters.get(root1, {word1})
        cluster2 = clusters.get(root2, {word2})
        if root1 != root2:
            compatible = all(
                best_scores.get(tuple(sorted((left, right))), 0.0) >= cohesion_threshold
                for left in cluster1
                for right in cluster2
            )
            if not compatible:
                continue
            merged = cluster1 | cluster2
            new_root = min(merged)
            clusters.pop(root1, None)
            clusters.pop(root2, None)
            clusters[new_root] = merged
            for word in merged:
                membership[word] = new_root
        retained.add((word1, word2, score))
    return sorted(retained, key=lambda item: (item[0], item[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-confidence", type=float, default=0.5)
    parser.add_argument("--cohesion-threshold", type=float, default=0.75)
    args = parser.parse_args()
    sample = [("alpha", "alfa", 0.95), ("alfa", "omega", 0.88), ("alpha", "omega", 0.2)]
    print(
        resolve_variant_pairs(
            sample, args.minimum_confidence, args.cohesion_threshold
        )
    )
    return 0


if __name__ == "__main__":
    resolved = resolve_variant_pairs(
        [("alpha", "alfa", 0.95), ("alfa", "omega", 0.88), ("alpha", "omega", 0.2)]
    )
    assert len(resolved) == 1
    raise SystemExit(main())
