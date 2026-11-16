"""
ingestion.py
============
Loads raw flight data into Spark DataFrames using an explicit StructType schema,
enforcing type safety, strict parsing, and schema validation.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
)

from src.spark_session import get_spark_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Ingestion")

# Explicit US DOT / Kaggle Flight Schema
FLIGHTS_RAW_SCHEMA = StructType([
    StructField("FL_DATE", StringType(), False),
    StructField("OP_CARRIER", StringType(), False),
    StructField("OP_CARRIER_FL_NUM", IntegerType(), True),
    StructField("ORIGIN", StringType(), False),
    StructField("DEST", StringType(), False),
    StructField("CRS_DEP_TIME", IntegerType(), False),
    StructField("DEP_TIME", DoubleType(), True),
    StructField("DEP_DELAY", DoubleType(), True),
    StructField("CRS_ARR_TIME", IntegerType(), False),
    StructField("ARR_TIME", DoubleType(), True),
    StructField("ARR_DELAY", DoubleType(), True),
    StructField("CANCELLED", DoubleType(), False),
    StructField("CANCELLATION_CODE", StringType(), True),
    StructField("DIVERTED", DoubleType(), False),
    StructField("CRS_ELAPSED_TIME", DoubleType(), True),
    StructField("ACTUAL_ELAPSED_TIME", DoubleType(), True),
    StructField("AIR_TIME", DoubleType(), True),
    StructField("DISTANCE", DoubleType(), False),
    StructField("CARRIER_DELAY", DoubleType(), True),
    StructField("WEATHER_DELAY", DoubleType(), True),
    StructField("NAS_DELAY", DoubleType(), True),
    StructField("SECURITY_DELAY", DoubleType(), True),
    StructField("LATE_AIRCRAFT_DELAY", DoubleType(), True),
])


def load_raw_flight_data(spark: SparkSession, csv_path: str = "data/raw/flights_raw.csv") -> DataFrame:
    """
    Loads raw CSV data into a PySpark DataFrame with explicit schema.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session.
    csv_path : str
        Path to the raw flights CSV.

    Returns
    -------
    DataFrame
        PySpark DataFrame adhering strictly to FLIGHTS_RAW_SCHEMA.
    """
    logger.info("Loading raw flight data from: %s", csv_path)

    df = (
        spark.read
        .format("csv")
        .option("header", "true")
        .option("mode", "DROPMALFORMED")
        .schema(FLIGHTS_RAW_SCHEMA)
        .load(csv_path)
    )

    row_count = df.count()
    logger.info("Ingestion completed successfully. Total rows loaded: %d, Columns: %d", row_count, len(df.columns))
    return df


if __name__ == "__main__":
    spark = get_spark_session("FlightDataIngestion")
    raw_df = load_raw_flight_data(spark)
    print("\n--- SCHEMA VERIFICATION ---")
    raw_df.printSchema()
    print("\n--- SAMPLE RECORDS (FIRST 5) ---")
    raw_df.select("FL_DATE", "OP_CARRIER", "ORIGIN", "DEST", "CRS_DEP_TIME", "DEP_DELAY", "ARR_DELAY", "CANCELLED").show(5)
    spark.stop()
