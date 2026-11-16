"""
preprocessing.py
================
Cleans raw flight data and conducts advanced feature engineering using:
- Spark DataFrame API
- Spark SQL Queries (registered temp views)
- Spark Window Functions (rolling traffic and cumulative departures)
- GroupBy aggregations (carrier and route delay benchmarks)

Saves the engineered dataset in partitioned Parquet format for fast ML training.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import time
import logging
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

from src.spark_session import get_spark_session
from src.ingestion import load_raw_flight_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Preprocessing")


def clean_flight_data(df: DataFrame) -> DataFrame:
    """
    Cleans raw data by removing cancelled and diverted flights,
    filtering missing arrival delays, and imputing or sanitizing fields.
    """
    logger.info("Cleaning raw flight data (initial count: %d)...", df.count())

    # 1. Standard FAA filtering: Remove cancelled and diverted flights
    cleaned_df = df.filter((F.col("CANCELLED") == 0.0) & (F.col("DIVERTED") == 0.0))

    # 2. Filter records where target ARR_DELAY or key operational columns are null
    cleaned_df = cleaned_df.filter(
        F.col("ARR_DELAY").isNotNull() &
        F.col("CRS_DEP_TIME").isNotNull() &
        F.col("ORIGIN").isNotNull() &
        F.col("DEST").isNotNull() &
        F.col("OP_CARRIER").isNotNull() &
        F.col("DISTANCE").isNotNull()
    )

    # 3. Create target label: 1.0 if ARR_DELAY > 15.0 minutes, else 0.0
    cleaned_df = cleaned_df.withColumn(
        "is_delayed",
        F.when(F.col("ARR_DELAY") > 15.0, 1.0).otherwise(0.0).cast(DoubleType())
    )

    logger.info("Cleaned records count: %d", cleaned_df.count())
    return cleaned_df


def engineer_features(spark: SparkSession, df: DataFrame) -> DataFrame:
    """
    Applies comprehensive feature engineering:
    - Temporal features (hour, day of week, month, season, time-of-day bucket)
    - Spark SQL queries for route & carrier benchmarks
    - Spark Window functions for airport congestion and cumulative volume
    - Distance categorization
    """
    logger.info("Starting feature engineering pipeline...")
    start_time = time.time()

    # 1. Temporal feature extraction
    # CRS_DEP_TIME is in HHMM format (e.g. 830 -> 8, 1745 -> 17)
    df_temporal = df.withColumn(
        "CRS_DEP_HOUR",
        F.floor(F.col("CRS_DEP_TIME") / 100).cast(IntegerType())
    ).withColumn(
        "CRS_DEP_MINUTE",
        (F.col("CRS_DEP_TIME") % 100).cast(IntegerType())
    )

    # Parse FL_DATE string to date type
    df_temporal = df_temporal.withColumn("flight_date", F.to_date(F.col("FL_DATE"), "yyyy-MM-dd"))
    df_temporal = df_temporal.withColumn("DAY_OF_WEEK", F.dayofweek(F.col("flight_date")).cast(IntegerType())) # 1=Sun, 7=Sat
    df_temporal = df_temporal.withColumn("DAY_OF_MONTH", F.dayofmonth(F.col("flight_date")).cast(IntegerType()))
    df_temporal = df_temporal.withColumn("MONTH", F.month(F.col("flight_date")).cast(IntegerType()))

    # Time of Day Buckets
    df_temporal = df_temporal.withColumn(
        "DEP_TIME_OF_DAY",
        F.when((F.col("CRS_DEP_HOUR") >= 5) & (F.col("CRS_DEP_HOUR") < 12), "Morning")
        .when((F.col("CRS_DEP_HOUR") >= 12) & (F.col("CRS_DEP_HOUR") < 17), "Afternoon")
        .when((F.col("CRS_DEP_HOUR") >= 17) & (F.col("CRS_DEP_HOUR") < 21), "Evening")
        .otherwise("Night")
    )

    # Seasonality
    df_temporal = df_temporal.withColumn(
        "SEASON",
        F.when(F.col("MONTH").isin(12, 1, 2), "Winter")
        .when(F.col("MONTH").isin(3, 4, 5), "Spring")
        .when(F.col("MONTH").isin(6, 7, 8), "Summer")
        .otherwise("Fall")
    )

    # Route definition and Distance Buckets
    df_temporal = df_temporal.withColumn(
        "ROUTE",
        F.concat(F.col("ORIGIN"), F.lit("-"), F.col("DEST"))
    ).withColumn(
        "DISTANCE_GROUP",
        F.when(F.col("DISTANCE") < 500.0, "ShortHaul")
        .when(F.col("DISTANCE") <= 1500.0, "MediumHaul")
        .otherwise("LongHaul")
    )

    # 2. Spark SQL Query: Compute carrier historical delay rate and volume benchmarks
    logger.info("Registering temporary view 'flights_temporal' and executing Spark SQL...")
    df_temporal.createOrReplaceTempView("flights_temporal")

    carrier_stats_sql = """
        SELECT
            OP_CARRIER,
            ROUND(AVG(is_delayed), 4) AS CARRIER_DELAY_RATE,
            COUNT(*) AS CARRIER_TOTAL_FLIGHTS
        FROM flights_temporal
        GROUP BY OP_CARRIER
    """
    carrier_stats_df = spark.sql(carrier_stats_sql)
    carrier_stats_df.createOrReplaceTempView("carrier_stats")

    # Join carrier statistics back via SQL
    joined_sql = """
        SELECT
            f.*,
            c.CARRIER_DELAY_RATE,
            c.CARRIER_TOTAL_FLIGHTS
        FROM flights_temporal f
        LEFT JOIN carrier_stats c
        ON f.OP_CARRIER = c.OP_CARRIER
    """
    df_sql = spark.sql(joined_sql)

    # 3. Spark Window Functions
    logger.info("Applying Spark Window functions for airport congestion tracking...")

    # Window A: Hourly departure congestion at origin airport on that date
    orig_hourly_window = Window.partitionBy("ORIGIN", "FL_DATE", "CRS_DEP_HOUR")
    df_window = df_sql.withColumn(
        "ORIGIN_HOURLY_TRAFFIC",
        F.count("*").over(orig_hourly_window).cast(DoubleType())
    )

    # Window B: Hourly arrival traffic at destination airport on that date
    dest_hourly_window = Window.partitionBy("DEST", "FL_DATE", "CRS_DEP_HOUR")
    df_window = df_window.withColumn(
        "DEST_HOURLY_TRAFFIC",
        F.count("*").over(dest_hourly_window).cast(DoubleType())
    )

    # Window C: Cumulative departures from origin airport today up to scheduled departure
    orig_cum_window = Window.partitionBy("ORIGIN", "FL_DATE").orderBy("CRS_DEP_TIME").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    df_window = df_window.withColumn(
        "ORIGIN_DAILY_CUMULATIVE_FLIGHTS",
        F.count("*").over(orig_cum_window).cast(DoubleType())
    )

    # 4. Select finalized modeling features and target
    modeling_columns = [
        # Target
        "is_delayed",
        # Categorical features
        "OP_CARRIER",
        "ORIGIN",
        "DEST",
        "DEP_TIME_OF_DAY",
        "SEASON",
        "DISTANCE_GROUP",
        # Numerical features
        "CRS_DEP_HOUR",
        "DAY_OF_WEEK",
        "DAY_OF_MONTH",
        "MONTH",
        "DISTANCE",
        "CRS_ELAPSED_TIME",
        "CARRIER_DELAY_RATE",
        "ORIGIN_HOURLY_TRAFFIC",
        "DEST_HOURLY_TRAFFIC",
        "ORIGIN_DAILY_CUMULATIVE_FLIGHTS"
    ]

    final_df = df_window.select(*modeling_columns)
    elapsed = time.time() - start_time
    logger.info("Feature engineering completed in %.2f seconds.", elapsed)
    return final_df


def save_processed_parquet(df: DataFrame, output_path: str = "data/processed/flights_processed.parquet"):
    """
    Saves the processed DataFrame in Parquet format, partitioned by MONTH for optimal query performance.
    """
    logger.info("Saving processed dataset to Parquet: %s ...", output_path)
    os.makedirs(output_path, exist_ok=True)

    (
        df.write
        .mode("overwrite")
        .partitionBy("MONTH")
        .parquet(output_path)
    )
    logger.info("Successfully persisted Parquet dataset to %s", output_path)


def run_preprocessing_pipeline(
    raw_csv: str = "data/raw/flights_raw.csv",
    output_parquet: str = "data/processed/flights_processed.parquet"
) -> DataFrame:
    """
    End-to-end preprocessing pipeline coordinator.
    """
    spark = get_spark_session("FlightPreprocessingPipeline")
    raw_df = load_raw_flight_data(spark, raw_csv)
    cleaned_df = clean_flight_data(raw_df)
    features_df = engineer_features(spark, cleaned_df)

    save_processed_parquet(features_df, output_parquet)

    # Read back to verify
    saved_df = spark.read.parquet(output_parquet)
    logger.info("Verified saved Parquet record count: %d", saved_df.count())
    logger.info("Class distribution of is_delayed:")
    saved_df.groupBy("is_delayed").count().show()

    return saved_df


if __name__ == "__main__":
    run_preprocessing_pipeline()
