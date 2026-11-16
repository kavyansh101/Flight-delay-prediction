"""
spark_session.py
================
Centralized SparkSession factory configured for reliable local execution across
Windows, macOS, and Linux platforms.
"""

import os
import sys
import logging
from pyspark.sql import SparkSession

# Ensure worker processes use the exact same active Python interpreter
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SparkSessionManager")


def get_spark_session(
    app_name: str = "FlightDelayPrediction",
    master: str = "local[*]",
    driver_memory: str = "4g",
    shuffle_partitions: int = 8,
) -> SparkSession:
    """
    Creates or retrieves an optimized local PySpark SparkSession.

    Parameters
    ----------
    app_name : str
        Name of the Spark application.
    master : str
        Master URL (default is local[*] to utilize all available CPU cores).
    driver_memory : str
        Memory allocated to Spark Driver (default: '4g').
    shuffle_partitions : int
        Number of shuffle partitions (default: 8, optimized for multi-core laptop).

    Returns
    -------
    SparkSession
        Configured and active SparkSession instance.
    """
    logger.info("Initializing SparkSession: %s (master=%s, memory=%s)", app_name, master, driver_memory)

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", driver_memory)
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.default.parallelism", str(shuffle_partitions))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    logger.info("SparkSession initialized successfully. Spark version: %s", spark.version)
    return spark


if __name__ == "__main__":
    spark = get_spark_session("SparkSessionTest")
    print(f"Spark Version: {spark.version}")
    print(f"Python Executable: {sys.executable}")
    test_df = spark.createDataFrame([(1, "Spark Local Mode Active")], ["id", "status"])
    test_df.show(truncate=False)
    spark.stop()
