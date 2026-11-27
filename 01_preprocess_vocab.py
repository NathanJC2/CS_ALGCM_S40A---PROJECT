"""Read text or CSV corpora and produce normalized words and sentence contexts."""

from __future__ import annotations

import argparse
import csv
import re
from html import unescape
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\r?\n+")
_WORD = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


def tokenize_text(text: str) -> list[str]:
    """Return lowercased Unicode word tokens, preserving internal hyphens/apostrophes."""
    cleaned = unescape(str(text))
    pattern = re.compile(r"[^\W\d_]+(?:[-\u2019'][^\W\d_]+)*", flags=re.UNICODE)
    return [match.group(0).casefold() for match in pattern.finditer(cleaned)]


def tokenize_sentences(text: str) -> list[list[str]]:
    """Split text into sentence-like units and tokenize each non-empty unit."""
    return [
        tokens
        for segment in _SENTENCE_BOUNDARY.split(text)
        if (tokens := tokenize_text(segment))
    ]


def read_corpus(
    paths: Sequence[str | Path], encoding: str = "utf-8"
) -> tuple[Counter[str], list[list[str]]]:
    """Read .txt/.csv files and return exact token counts and tokenized sentences.

    CSV rows are treated as text by joining their cells with spaces. The caller
    controls all corpus paths; unsupported extensions are rejected explicitly.
    """
    if not paths:
        raise ValueError("Provide at least one .txt or .csv corpus path.")

    sentences: list[list[str]] = []
    for raw_path in paths:
        path = Path(raw_path)
        suffix = path.suffix.lower()
        if suffix == ".txt":
            text = path.read_text(encoding=encoding, errors="replace")
            sentences.extend(tokenize_sentences(text))
        elif suffix == ".csv":
            with path.open("r", encoding=encoding, errors="replace", newline="") as stream:
                for row in csv.reader(stream):
                    sentences.extend(tokenize_sentences(" ".join(row)))
        else:
            raise ValueError(f"Unsupported corpus extension for {path.name!r}.")

    counts: Counter[str] = Counter(
        word for sentence in sentences for word in sentence
    )
    return counts, sentences


def tag_vocabulary(
    counts: Mapping[str, int],
    batch_size: int = 1000,
) -> dict[str, dict[str, object]]:
    """Predict PinoyBot labels for unique vocabulary words in batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    try:
        from pinoybot import predict_language
    except ImportError as error:
        raise RuntimeError("PinoyBot integration is unavailable") from error

    words = sorted(counts)
    predictions: dict[str, dict[str, object]] = {}
    for start in range(0, len(words), batch_size):
        batch = words[start:start + batch_size]
        labels, probabilities, classes = predict_language(batch)
        for word, label, scores in zip(batch, labels, probabilities):
            score_map = {name: float(score) for name, score in zip(classes, scores)}
            predictions[word] = {
                "label": str(label),
                "probabilities": score_map,
                "confidence": max(score_map.values(), default=0.0),
            }
    return predictions


def main(argv: Iterable[str] | None = None) -> int:
    """Command-line smoke runner for corpus preprocessing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Input .txt and/or .csv files")
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args(argv)
    counts, sentences = read_corpus(args.paths, encoding=args.encoding)
    print(f"tokens={sum(counts.values())} vocabulary={len(counts)} sentences={len(sentences)}")
    return 0


if __name__ == "__main__":
    assert tokenize_text("Hello, WORLD! 123") == ["hello", "world"]
    assert tokenize_sentences("One word. Two words!") == [
        ["one", "word"],
        ["two", "words"],
    ]
    raise SystemExit(main())
