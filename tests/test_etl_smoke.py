from bank_marketing.etl.spark_utils import get_spark
from bank_marketing.etl.schemas import BANK_SCHEMA, BANK_COLUMNS, BANK_SCHEMA_DDL
from pyspark.sql import functions as F

def test_schema_fields_count():
    assert len(BANK_COLUMNS) == 17
    assert len(BANK_SCHEMA.fields) == 17

def test_from_csv_parses_after_cleaning():
    spark = get_spark("etl_smoke")
    sample = ['"58;""management"";""married"";""tertiary"";""no"";2143;""yes"";""no"";""unknown"";5;""may"";261;1;-1;0;""unknown"";""no"""']
    df = spark.createDataFrame(sample, "string").toDF("value")
    line = F.regexp_replace(F.col("value"), r'^\s*"', "")
    line = F.regexp_replace(line, r'"\s*$', "")
    line = F.regexp_replace(line, r'""', '"')
    csv_opts = {"sep": ";", "quote": '"'}
    parsed = df.select(F.from_csv(line, BANK_SCHEMA_DDL, csv_opts).alias("r")).select("r.*")
    assert parsed.count() == 1
    row = parsed.first()
    assert row.age == 58 and row.job == "management"
