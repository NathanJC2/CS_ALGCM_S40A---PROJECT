"""Export canonicalized, deduplicated variant pairs with exact CSV headers."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Iterable, Sequence


CSV_COLUMNS = (
    "word1",
    "word2",
    "word1_count",
    "word2_count",
    "character_difference",
)


def _load_edit_analyzer():
    module_path = Path(__file__).with_name("03_dp_edit_distance.py")
    spec = importlib.util.spec_from_file_location("variant_export_edit_distance", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load edit-distance module at {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.analyze_edit_distance


_analyze_edit_distance = _load_edit_analyzer()


def export_variants(
    weighted_pairs: Iterable[Sequence[object]],
    frequencies: dict[str, int],
    output_path: str | Path,
) -> int:
    """Write pairs alphabetically canonicalized and deduplicated to ``output_path``."""
    unique_pairs: set[tuple[str, str]] = set()
    for item in weighted_pairs:
        if len(item) < 2:
            raise ValueError("Each pair must contain at least two words.")
        word1, word2 = str(item[0]), str(item[1])
        if word1 != word2:
            unique_pairs.add(tuple(sorted((word1, word2))))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for word1, word2 in sorted(unique_pairs):
            analysis = _analyze_edit_distance(word1, word2)
            writer.writerow(
                {
                    "word1": word1,
                    "word2": word2,
                    "word1_count": frequencies.get(word1, 0),
                    "word2_count": frequencies.get(word2, 0),
                    "character_difference": analysis.character_difference,
                }
            )
    return len(unique_pairs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", required=True, help="Destination CSV path")
    args = parser.parse_args()
    count = export_variants(
        [("pulitiko", "politiko", 0.9), ("politiko", "pulitiko", 0.8)],
        {"politiko": 4, "pulitiko": 3},
        args.output_csv,
    )
    print(f"exported_pairs={count}")
    return 0


if __name__ == "__main__":
    assert CSV_COLUMNS == (
        "word1", "word2", "word1_count", "word2_count", "character_difference"
    )
    raise SystemExit(main())
