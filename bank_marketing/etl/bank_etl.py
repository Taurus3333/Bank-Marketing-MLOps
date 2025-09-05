from __future__ import annotations

import argparse
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import StringType

from ..core.logging import get_logger
from ..core.exceptions import AppError, ensure
from ..config import DATA_DIR
from .spark_utils import get_spark
from .schemas import BANK_SCHEMA, BANK_COLUMNS, BANK_SCHEMA_DDL
from .validators import require_columns, validate_allowed_values


def _clean_and_parse(path: str) -> DataFrame:
    """
    Handles the unusual quoting:
      - Each line wrapped in double quotes
      - Inner quotes doubled (""), delimiter is ';'
    Strategy: read as text -> clean quotes -> parse with from_csv
    """
    spark = get_spark()

    raw = spark.read.text(path)  # column: 'value'

    # 1) Strip a single leading/trailing quote if present
    line = F.regexp_replace(F.col("value"), r'^\s*"', "")
    line = F.regexp_replace(line, r'"\s*$', "")

    # 2) Collapse doubled quotes to a single quote
    line = F.regexp_replace(line, r'""', '"').alias("line")

    clean = raw.select(line)

    # 3) Drop header row (starts with 'age;')
    clean = clean.filter(~F.lower(F.col("line")).startswith("age;"))

    

    # 4) Parse strictly with from_csv into the fixed schema (DDL string)
    csv_opts = {"sep": ";", "quote": '"'}
    parsed = clean.select(F.from_csv(F.col("line"), BANK_SCHEMA_DDL, csv_opts).alias("r"))
    df = parsed.select("r.*")


    # 5) Lowercase & trim string columns; enforce column order
    for c, t in df.dtypes:
        if t == "string":
            df = df.withColumn(c, F.trim(F.lower(F.col(c))).cast(StringType()))
    df = df.select(BANK_COLUMNS)  # whitelist & order

    return df


def _basic_integrity(df: DataFrame) -> DataFrame:
    # No missing attributes per spec; still assert.
    for c in df.columns:
        nulls = df.where(F.col(c).isNull()).limit(1).count()
        ensure(nulls == 0, f"Nulls found in column '{c}'")

    # Categorical domain checks
    validate_allowed_values(df)

    # Numeric sanity (duration >= 0, pdays >= -1, etc.)
    bad = (
        df.where(
            (F.col("age") < 0) |
            (F.col("duration") < 0) |
            (F.col("campaign") < 0) |
            (F.col("pdays") < -1) |
            (F.col("previous") < 0)
        )
        .limit(1)
        .count()
    )
    ensure(bad == 0, "Numeric sanity checks failed")

    return df


def run(input_path: str, output_dir: str) -> None:
    log = get_logger(__name__)
    log.info(f"ETL start: input={input_path}, output_dir={output_dir}")

    df = _clean_and_parse(input_path)
    require_columns(df)
    df = _basic_integrity(df)

    # Write Parquet (atomic overwrite), splittable for scale
    (
        df.write
          .mode("overwrite")
          .parquet(output_dir)
    )
    log.info("ETL success")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Bank Marketing ETL (lean)")
    parser.add_argument("--input", required=False,
                        default=str((DATA_DIR / "bank.csv").as_posix()))
    parser.add_argument("--output", required=False,
                        default=str((DATA_DIR / "clean" / "bank.parquet").as_posix()))
    args = parser.parse_args()
    try:
        run(args.input, args.output)
    except AppError as e:
        get_logger(__name__).error(f"ETL failed: {e}")
        raise
    except Exception as e:
        # Wrap unexpected errors for consistent observability
        raise AppError("Unhandled ETL error", cause=e) from e


if __name__ == "__main__":
    cli()
