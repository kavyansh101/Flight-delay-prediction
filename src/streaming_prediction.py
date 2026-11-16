"""
streaming_prediction.py
=======================
Demonstrates Spark Structured Streaming reading live flight records from
data/streaming_input/, applying the saved MLlib pipeline model in real time,
and outputting live delay alert predictions.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import time
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
)
from pyspark.ml.pipeline import PipelineModel

from src.spark_session import get_spark_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("StreamingPrediction")

STREAMING_INPUT_DIR = "data/streaming_input"
STREAMING_OUTPUT_DIR = "data/streaming_output"
CHECKPOINT_DIR = "data/streaming_checkpoint"
MODEL_PATH = "models/best_flight_delay_model"

STREAM_SCHEMA = StructType([
    StructField("flight_id", StringType(), False),
    StructField("timestamp", StringType(), False),
    StructField("is_delayed", DoubleType(), True),
    StructField("OP_CARRIER", StringType(), False),
    StructField("ORIGIN", StringType(), False),
    StructField("DEST", StringType(), False),
    StructField("DEP_TIME_OF_DAY", StringType(), False),
    StructField("SEASON", StringType(), False),
    StructField("DISTANCE_GROUP", StringType(), False),
    StructField("CRS_DEP_HOUR", IntegerType(), False),
    StructField("DAY_OF_WEEK", IntegerType(), False),
    StructField("DAY_OF_MONTH", IntegerType(), False),
    StructField("MONTH", IntegerType(), False),
    StructField("DISTANCE", DoubleType(), False),
    StructField("CRS_ELAPSED_TIME", DoubleType(), False),
    StructField("CARRIER_DELAY_RATE", DoubleType(), False),
    StructField("ORIGIN_HOURLY_TRAFFIC", DoubleType(), False),
    StructField("DEST_HOURLY_TRAFFIC", DoubleType(), False),
    StructField("ORIGIN_DAILY_CUMULATIVE_FLIGHTS", DoubleType(), False),
    StructField("class_weight", DoubleType(), True),
])


def process_micro_batch(batch_df, batch_id, model: PipelineModel):
    """
    Called by foreachBatch on every incoming micro-batch.
    Applies the MLlib pipeline and displays real-time delay alert predictions.
    """
    if batch_df.isEmpty():
        return

    logger.info("Processing Streaming Micro-Batch ID: %d (%d records)...", batch_id, batch_df.count())

    # Apply MLlib pipeline model
    predictions = model.transform(batch_df)

    # Format display columns
    scored_df = predictions.select(
        "flight_id",
        "OP_CARRIER",
        "ORIGIN",
        "DEST",
        "CRS_DEP_HOUR",
        "DISTANCE",
        "prediction"
    ).withColumn(
        "Predicted_Status",
        F.when(F.col("prediction") == 1.0, "[DELAYED > 15m]").otherwise("[ON-TIME]")
    )

    print(f"\n==================== STREAMING MICRO-BATCH {batch_id} PREDICTIONS ====================")
    scored_df.show(15, truncate=False)

    # Write output to streaming_output sink
    out_path = os.path.join(STREAMING_OUTPUT_DIR, f"batch_{batch_id}.parquet")
    scored_df.write.mode("overwrite").parquet(out_path)
    logger.info("Micro-Batch %d predictions saved to %s", batch_id, out_path)


def run_streaming_pipeline(max_batches: int = 2):
    """
    Configures and launches the Spark Structured Streaming pipeline.
    """
    os.makedirs(STREAMING_INPUT_DIR, exist_ok=True)
    os.makedirs(STREAMING_OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    spark = get_spark_session("FlightDelayStructuredStreaming")
    logger.info("Loading MLlib PipelineModel from: %s", MODEL_PATH)
    model = PipelineModel.load(MODEL_PATH)

    logger.info("Setting up Structured Streaming readStream on: %s", STREAMING_INPUT_DIR)
    stream_df = (
        spark.readStream
        .format("json")
        .schema(STREAM_SCHEMA)
        .option("maxFilesPerTrigger", 1)  # Process 1 file per trigger for clear micro-batches
        .load(STREAMING_INPUT_DIR)
    )

    query = (
        stream_df.writeStream
        .foreachBatch(lambda df, b_id: process_micro_batch(df, b_id, model))
        .option("checkpointLocation", CHECKPOINT_DIR)
        .trigger(processingTime="2 seconds")
        .start()
    )

    logger.info("Streaming query started. Awaiting micro-batches...")
    # Run for a demonstration duration or until stopped
    try:
        # Wait for processing
        query.awaitTermination(timeout=20)
    except Exception as e:
        logger.info("Streaming finished: %s", e)
    finally:
        query.stop()
        spark.stop()
        logger.info("Structured Streaming session stopped.")


if __name__ == "__main__":
    run_streaming_pipeline()
