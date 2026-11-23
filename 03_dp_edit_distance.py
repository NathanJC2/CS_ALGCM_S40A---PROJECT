"""Wagner-Fischer edit distance with aligned character edits and positions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class EditAnalysis:
    """Edit distance and the ordered, aligned character-level changes."""

    distance: int
    transformations: tuple[str, ...]
    normalized_positions: tuple[float, ...]

    @property
    def character_difference(self) -> str:
        """Return edits in the requested ``character vs. character`` format."""
        return "; ".join(self.transformations) if self.transformations else "identical"


def analyze_edit_distance(source: str, target: str) -> EditAnalysis:
    """Compute Levenshtein distance and backtrace exact insert/delete/substitute edits.

    Edit positions use the changed source/target character index normalized to
    the inclusive range 0.0--1.0; an insertion at the end therefore has
    position 1.0.
    """
    rows, columns = len(source) + 1, len(target) + 1
    costs = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        costs[row][0] = row
    for column in range(columns):
        costs[0][column] = column

    for row in range(1, rows):
        for column in range(1, columns):
            substitution_cost = 0 if source[row - 1] == target[column - 1] else 1
            costs[row][column] = min(
                costs[row - 1][column] + 1,
                costs[row][column - 1] + 1,
                costs[row - 1][column - 1] + substitution_cost,
            )

    edits: list[tuple[str, float]] = []
    row, column = len(source), len(target)
    position_denominator = max(len(source), len(target), 1) - 1
    while row or column:
        if (
            row
            and column
            and source[row - 1] == target[column - 1]
            and costs[row][column] == costs[row - 1][column - 1]
        ):
            row -= 1
            column -= 1
            continue
        if row and column and costs[row][column] == costs[row - 1][column - 1] + 1:
            description = f"{source[row - 1]} vs. {target[column - 1]}"
            position_index = max(row - 1, column - 1)
            row -= 1
            column -= 1
        elif row and costs[row][column] == costs[row - 1][column] + 1:
            description = f"{source[row - 1]} vs. -"
            position_index = row - 1
            row -= 1
        else:
            description = f"- vs. {target[column - 1]}"
            position_index = column - 1
            column -= 1
        normalized = (
            min(1.0, max(0.0, position_index / position_denominator))
            if position_denominator
            else 0.0
        )
        edits.append((description, normalized))

    edits.reverse()
    return EditAnalysis(
        distance=costs[-1][-1],
        transformations=tuple(item[0] for item in edits),
        normalized_positions=tuple(item[1] for item in edits),
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default="politiko")
    parser.add_argument("target", nargs="?", default="pulitiko")
    args = parser.parse_args(argv)
    analysis = analyze_edit_distance(args.source, args.target)
    print(
        f"distance={analysis.distance} "
        f"transformations={analysis.character_difference!r} "
        f"positions={analysis.normalized_positions}"
    )
    return 0


if __name__ == "__main__":
    sample = analyze_edit_distance("cat", "cut")
    assert sample.distance == 1 and sample.transformations == ("a vs. u",)
    assert analyze_edit_distance("at", "cat").transformations == ("- vs. c",)
    raise SystemExit(main())
