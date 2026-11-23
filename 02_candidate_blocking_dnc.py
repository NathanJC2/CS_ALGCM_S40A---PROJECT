"""Divide-and-conquer candidate blocking using shared character signatures."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import islice, product
from typing import Iterable, Sequence


def _signatures(word: str) -> frozenset[str]:
    """Build lightweight unigram, bigram, and consonant-skeleton signatures."""
    lowered = word.lower()
    consonants = "".join(char for char in lowered if char not in "aeiou")
    signatures = {f"u:{char}" for char in lowered}
    signatures.update(f"b:{lowered[index:index + 2]}" for index in range(len(lowered) - 1))
    signatures.add(f"s:{consonants}")
    return frozenset(signatures)


def _signature_index(words: Sequence[str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for word in words:
        for signature in _signatures(word):
            index[signature].append(word)
    return index


def _add_cross_pairs(
    left: dict[str, list[str]],
    right: dict[str, list[str]],
    output: set[tuple[str, str]],
    max_pairs_per_signature: int,
) -> None:
    for signature in left.keys() & right.keys():
        pairs = product(left[signature], right[signature])
        for word1, word2 in islice(pairs, max_pairs_per_signature):
            if word1 != word2:
                output.add(tuple(sorted((word1, word2))))


def _divide_bucket(
    words: Sequence[str],
    output: set[tuple[str, str]],
    leaf_size: int,
    max_pairs_per_signature: int,
) -> dict[str, list[str]]:
    """Recursively split a length bucket and join only shared signatures."""
    if len(words) <= leaf_size:
        index = _signature_index(words)
        _add_cross_pairs(index, index, output, max_pairs_per_signature)
        return index

    midpoint = len(words) // 2
    left = _divide_bucket(words[:midpoint], output, leaf_size, max_pairs_per_signature)
    right = _divide_bucket(words[midpoint:], output, leaf_size, max_pairs_per_signature)
    _add_cross_pairs(left, right, output, max_pairs_per_signature)

    merged: dict[str, list[str]] = defaultdict(list)
    for index in (left, right):
        for signature, values in index.items():
            merged[signature].extend(values)
    return merged


def generate_candidates(
    vocabulary: Iterable[str],
    max_length_diff: int = 1,
    leaf_size: int = 64,
    max_pairs_per_signature: int = 5000,
) -> list[tuple[str, str]]:
    """Generate deduplicated candidate pairs from adjacent length buckets.

    Candidate comparisons are performed through bounded signature postings,
    rather than an exhaustive vocabulary-wide Cartesian product. This is a
    blocking heuristic: downstream context and classifier stages decide whether
    visually similar words are actual spelling variants.
    """
    if max_length_diff < 0 or leaf_size < 1 or max_pairs_per_signature < 1:
        raise ValueError("Blocking limits must be positive (length difference may be zero).")

    buckets: dict[int, list[str]] = defaultdict(list)
    for word in sorted(set(vocabulary)):
        if word:
            buckets[len(word)].append(word)

    output: set[tuple[str, str]] = set()
    indexes: dict[int, dict[str, list[str]]] = {}
    for length, words in sorted(buckets.items()):
        indexes[length] = _divide_bucket(
            words, output, leaf_size, max_pairs_per_signature
        )

    lengths = sorted(indexes)
    for position, length in enumerate(lengths):
        for other_length in lengths[position + 1:]:
            if other_length - length > max_length_diff:
                break
            _add_cross_pairs(
                indexes[length],
                indexes[other_length],
                output,
                max_pairs_per_signature,
            )
    return sorted(output)


def main() -> int:
    """Run a small synthetic blocking example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("words", nargs="*", help="Vocabulary words")
    parser.add_argument("--max-length-diff", type=int, default=1)
    parser.add_argument("--leaf-size", type=int, default=64)
    parser.add_argument("--max-pairs-per-signature", type=int, default=5000)
    args = parser.parse_args()
    words = args.words or ["politiko", "pulitiko", "opo", "upo", "book"]
    for pair in generate_candidates(
        words, args.max_length_diff, args.leaf_size, args.max_pairs_per_signature
    ):
        print(pair)
    return 0


if __name__ == "__main__":
    assert ("politiko", "pulitiko") in generate_candidates(
        ["politiko", "pulitiko", "other"]
    )
    raise SystemExit(main())
