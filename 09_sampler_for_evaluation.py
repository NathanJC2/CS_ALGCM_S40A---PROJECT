"""Create a reproducible manual-evaluation workbook from an exported pair CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from openpyxl import load_workbook


REQUIRED_COLUMNS = (
    "word1",
    "word2",
    "word1_count",
    "word2_count",
    "character_difference",
)


def create_evaluation_sample(
    input_csv: str | Path,
    output_xlsx: str | Path,
    sample_size: int = 100,
    random_state: int = 42,
) -> int:
    """Sample rows reproducibly and add blank labels plus an Excel precision formula."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive.")
    variants = pd.read_csv(input_csv)
    missing = set(REQUIRED_COLUMNS) - set(variants.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    sampled = variants.sample(
        n=min(sample_size, len(variants)), random_state=random_state
    ).loc[:, list(REQUIRED_COLUMNS)].copy()
    sampled["is_correct"] = ""

    output_path = Path(output_xlsx)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_excel(output_path, index=False)
    workbook = load_workbook(output_path)
    worksheet = workbook.active
    worksheet["G1"] = "manual_evaluation_status"
    worksheet["G2"] = "UNAVAILABLE - enter TRUE/FALSE in is_correct"
    worksheet["H1"] = "overall_precision"
    last_row = max(2, len(sampled) + 1)
    worksheet["H2"] = (
        f'=IF(COUNTIF(F2:F{last_row},"TRUE")+COUNTIF(F2:F{last_row},"FALSE")=0,'
        f'"UNAVAILABLE",COUNTIF(F2:F{last_row},"TRUE")/'
        f'(COUNTIF(F2:F{last_row},"TRUE")+COUNTIF(F2:F{last_row},"FALSE")))'
    )
    workbook.save(output_path)
    return len(sampled)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-xlsx", required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)
    sampled = create_evaluation_sample(
        args.input_csv, args.output_xlsx, args.sample_size, args.random_state
    )
    print(f"sampled_pairs={sampled}")
    return 0


if __name__ == "__main__":
    assert len(REQUIRED_COLUMNS) == 5
    raise SystemExit(main())
