from __future__ import annotations
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType
)

BANK_SCHEMA = StructType([
    StructField("age", IntegerType(), False),
    StructField("job", StringType(), False),
    StructField("marital", StringType(), False),
    StructField("education", StringType(), False),
    StructField("default", StringType(), False),
    StructField("balance", IntegerType(), False),
    StructField("housing", StringType(), False),
    StructField("loan", StringType(), False),
    StructField("contact", StringType(), False),
    StructField("day", IntegerType(), False),
    StructField("month", StringType(), False),
    StructField("duration", IntegerType(), False),
    StructField("campaign", IntegerType(), False),
    StructField("pdays", IntegerType(), False),
    StructField("previous", IntegerType(), False),
    StructField("poutcome", StringType(), False),
    StructField("y", StringType(), False),
])

# Column whitelist (order preserved)
BANK_COLUMNS = [f.name for f in BANK_SCHEMA.fields]

# DDL string for from_csv (required on some Spark builds)
BANK_SCHEMA_DDL = (
    "age INT, "
    "job STRING, "
    "marital STRING, "
    "education STRING, "
    "default STRING, "
    "balance INT, "
    "housing STRING, "
    "loan STRING, "
    "contact STRING, "
    "day INT, "
    "month STRING, "
    "duration INT, "
    "campaign INT, "
    "pdays INT, "
    "previous INT, "
    "poutcome STRING, "
    "y STRING"
)

ALLOWED = {
    "job": {
        "admin.", "unknown", "unemployed", "management", "housemaid",
        "entrepreneur", "student", "blue-collar", "self-employed",
        "retired", "technician", "services"
    },
    "marital": {"married", "divorced", "single"},
    "education": {"unknown", "secondary", "primary", "tertiary"},
    "default": {"yes", "no"},
    "housing": {"yes", "no"},
    "loan": {"yes", "no"},
    "contact": {"unknown", "telephone", "cellular"},
    "month": {"jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"},
    "poutcome": {"unknown","other","failure","success"},
    "y": {"yes","no"},
}
