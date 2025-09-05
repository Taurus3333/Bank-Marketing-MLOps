from __future__ import annotations

import os
import sys
from pyspark.sql import SparkSession
from ..config import PYSPARK_APP_NAME



def get_spark(app_name: str = "app", master: str = "local[*]") -> SparkSession:
    # ensure PySpark uses the current Python interpreter (Windows friendly)
    python_exe = sys.executable  # path to current venv python
    os.environ["PYSPARK_PYTHON"] = python_exe
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_exe

    # optionally set SPARK_HOME if needed or other Spark configs
    spark = SparkSession.builder.master(master).appName(app_name).getOrCreate()
    # tune minimal local config if desired
    spark.conf.set("spark.sql.shuffle.partitions", "4")
    return spark

