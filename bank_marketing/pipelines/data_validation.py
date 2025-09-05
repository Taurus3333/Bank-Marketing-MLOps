from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd
from ..core.exceptions import AppError
from ..core.logging import get_logger
from ..etl.schemas import BANK_COLUMNS

log = get_logger(__name__)


def compute_hash(df: pd.DataFrame, subset_cols: list[str] | None = None) -> str:
    """
    Compute a stable sha1 hash of dataframe (sample-based for speed).
    Deterministic: sorts by columns and uses first N rows.
    """
    sub = df if subset_cols is None else df[subset_cols]
    # take head for large datasets
    head = sub.head(1000).copy()
    head = head.sort_values(list(head.columns)).reset_index(drop=True)
    txt = head.to_csv(index=False).encode("utf-8")
    return hashlib.sha1(txt).hexdigest()


def validate_schema(df: pd.DataFrame) -> None:
    got = [c.lower() for c in df.columns]
    expected = [c.lower() for c in BANK_COLUMNS]
    if not all(c in got for c in expected):
        missing = [c for c in expected if c not in got]
        raise AppError("Schema validation failed - missing expected columns", details={"missing": missing})
    log.info("Schema validation passed.")


def simple_validation_report(df: pd.DataFrame, out_path: str | Path) -> dict:
    """
    Produce a small validation report saved as JSON for reproducibility.
    """
    report = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "sample_hash": compute_hash(df),
        "class_counts": df["y"].value_counts().to_dict()
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info(f"Validation report written to {out_path}")
    return report
