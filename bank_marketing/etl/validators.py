from __future__ import annotations
from pyspark.sql import DataFrame, functions as F
from ..core.exceptions import AppError
from .schemas import BANK_COLUMNS, ALLOWED


def require_columns(df: DataFrame) -> None:
    got = [c.lower() for c in df.columns]
    exp = BANK_COLUMNS
    if got != exp:
        raise AppError(
            "Column mismatch",
            details={"expected": exp, "got": got}
        )


def validate_allowed_values(df: DataFrame) -> None:
    # For each categorical col with constraints, check invalids (case-normalized)
    problems: dict[str, int] = {}
    for col, allowed in ALLOWED.items():
        if df.schema[col].dataType.simpleString() != "string":
            continue
        invalid_cnt = (
            df.select(F.col(col))
              .where(~F.col(col).isin(list(allowed)))
              .limit(1)  # short-circuit to keep it lean
              .count()
        )
        if invalid_cnt > 0:
            problems[col] = invalid_cnt

    if problems:
        raise AppError("Found invalid categorical values", details=problems)
