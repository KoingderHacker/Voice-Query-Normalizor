"""Small shared helpers for the CLI, pipeline, and evaluation reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def vocabulary_metrics(original_vocabulary: set[str], transformed_vocabulary: set[str]) -> dict[str, int | float]:
    """Calculate morphology vocabulary sizes and complementary reduction/retention rates."""
    original_size = len(original_vocabulary)
    resulting_size = len(transformed_vocabulary)
    reduction = original_size - resulting_size
    if original_size == 0:
        reduction_percent = 0.0
        retention_percent = 0.0
    else:
        reduction_percent = 100.0 * reduction / original_size
        retention_percent = 100.0 * resulting_size / original_size
    return {
        "original_vocabulary_size": original_size,
        "resulting_vocabulary_size": resulting_size,
        "vocabulary_reduction": reduction,
        "vocabulary_reduction_percent": reduction_percent,
        "vocabulary_retention_percent": retention_percent,
    }


def write_csv(path: str | Path, fieldnames: Sequence[str], rows: Iterable[Mapping]) -> None:
    """Write a deterministic UTF-8 CSV, creating its parent directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
