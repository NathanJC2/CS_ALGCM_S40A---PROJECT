"""Read text or CSV corpora and produce normalized words and sentence contexts."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\r?\n+")
_WORD = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


def tokenize_text(text: str) -> list[str]:
    """Return lowercase alphabetic tokens, excluding numbers and punctuation."""
    return [match.group(0).lower() for match in _WORD.finditer(text)]


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
