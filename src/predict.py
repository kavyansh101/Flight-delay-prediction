"""
predict.py
==========
Loads the persisted Spark MLlib PipelineModel and performs inference on
single flight records or batch CSV input files.

Example CLI usage:
    # Single flight inference
    python src/predict.py --carrier AA --origin ORD --dest LAX --time 1830 --month 12 --day 15 --distance 1744

    # Default demo prediction mode
    python src/predict.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import argparse
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.ml.pipeline import PipelineModel

from src.spark_session import get_spark_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Predictor")

MODEL_PATH = "models/best_flight_delay_model"


def load_model(model_path: str = MODEL_PATH) -> PipelineModel:
    """Loads the fitted Spark MLlib PipelineModel."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at: {model_path}. Please run train.py first.")
    logger.info("Loading persisted MLlib PipelineModel from: %s", model_path)
    return PipelineModel.load(model_path)


def prepare_flight_features(
    spark: SparkSession,
    carrier: str,
    origin: str,
    dest: str,
    dep_time_hhmm: int,
    month: int,
    day_of_month: int,
    day_of_week: int,
    distance: float,
    carrier_delay_rate: float = 0.36,
    origin_hourly_traffic: float = 24.0,
    dest_hourly_traffic: float = 18.0,
    origin_daily_cumulative: float = 85.0
):
    """
    Constructs a Spark DataFrame with all required engineered features for inference.
    """
    dep_hour = int(dep_time_hhmm // 100)

    # Time of Day
    if 5 <= dep_hour < 12:
        tod = "Morning"
    elif 12 <= dep_hour < 17:
        tod = "Afternoon"
    elif 17 <= dep_hour < 21:
        tod = "Evening"
    else:
        tod = "Night"

    # Season
    if month in [12, 1, 2]:
        season = "Winter"
    elif month in [3, 4, 5]:
        season = "Spring"
    elif month in [6, 7, 8]:
        season = "Summer"
    else:
        season = "Fall"

    # Distance Group
    if distance < 500.0:
        dist_group = "ShortHaul"
    elif distance <= 1500.0:
        dist_group = "MediumHaul"
    else:
        dist_group = "LongHaul"

    crs_elapsed_time = round(45.0 + (distance / 460.0) * 60.0)

    data = [(
        0.0,  # dummy label
        carrier,
        origin,
        dest,
        tod,
        season,
        dist_group,
        int(dep_hour),
        int(day_of_week),
        int(day_of_month),
        int(month),
        float(distance),
        float(crs_elapsed_time),
        float(carrier_delay_rate),
        float(origin_hourly_traffic),
        float(dest_hourly_traffic),
        float(origin_daily_cumulative),
        1.0   # dummy class_weight
    )]

    schema = StructType([
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
        StructField("class_weight", DoubleType(), True)
    ])

    return spark.createDataFrame(data, schema)


def predict_flight(
    model: PipelineModel,
    flight_df
):
    """
    Transforms input DataFrame using the MLlib pipeline model and extracts prediction & probability.
    """
    predictions = model.transform(flight_df)
    res = predictions.select("prediction", "probability").collect()[0]

    pred_label = int(res["prediction"])
    prob_vector = res["probability"].toArray()
    delay_prob = float(prob_vector[1])  # Class 1: Delayed > 15 min
    ontime_prob = float(prob_vector[0])

    return pred_label, delay_prob, ontime_prob


def run_demo_predictions():
    """Runs predictions on 3 diverse test scenarios."""
    spark = get_spark_session("FlightDelayInferenceDemo")
    model = load_model(MODEL_PATH)

    test_cases = [
        {
            "desc": "Case 1: Morning Delta flight (MSP -> ATL, 07:15 AM, May)",
            "carrier": "DL", "origin": "MSP", "dest": "ATL",
            "time": 715, "month": 5, "day": 12, "dow": 2, "dist": 907.0,
            "carrier_rate": 0.28, "orig_traffic": 12.0, "dest_traffic": 32.0, "cum_flights": 18.0
        },
        {
            "desc": "Case 2: Friday Evening JetBlue holiday flight (JFK -> SFO, 19:45 PM, Dec)",
            "carrier": "B6", "origin": "JFK", "dest": "SFO",
            "time": 1945, "month": 12, "day": 20, "dow": 5, "dist": 2586.0,
            "carrier_rate": 0.44, "orig_traffic": 38.0, "dest_traffic": 28.0, "cum_flights": 165.0
        },
        {
            "desc": "Case 3: Summer Afternoon United flight (ORD -> DEN, 16:30 PM, July)",
            "carrier": "UA", "origin": "ORD", "dest": "DEN",
            "time": 1630, "month": 7, "day": 18, "dow": 4, "dist": 888.0,
            "carrier_rate": 0.38, "orig_traffic": 45.0, "dest_traffic": 34.0, "cum_flights": 120.0
        }
    ]

    print("\n" + "=" * 75)
    print("      FLIGHT DELAY PREDICTION INFERENCE DEMO (Apache Spark MLlib)      ")
    print("=" * 75)

    for case in test_cases:
        df = prepare_flight_features(
            spark,
            carrier=case["carrier"],
            origin=case["origin"],
            dest=case["dest"],
            dep_time_hhmm=case["time"],
            month=case["month"],
            day_of_month=case["day"],
            day_of_week=case["dow"],
            distance=case["dist"],
            carrier_delay_rate=case["carrier_rate"],
            origin_hourly_traffic=case["orig_traffic"],
            dest_hourly_traffic=case["dest_traffic"],
            origin_daily_cumulative=case["cum_flights"]
        )
        pred, delay_p, ontime_p = predict_flight(model, df)

        status = "[DELAYED > 15m]" if pred == 1 else "[ON-TIME]"
        risk = "HIGH" if delay_p > 0.55 else ("MODERATE" if delay_p > 0.40 else "LOW")

        print(f"\nScenario : {case['desc']}")
        print(f"Route    : {case['carrier']} | {case['origin']} -> {case['dest']} | Dist: {case['dist']} mi")
        print(f"Schedule : Dep: {case['time']:04d} hrs | Month: {case['month']} | Day: {case['day']}")
        print(f"Status   : {status}")
        print(f"Delay Prob: {delay_p*100:.1f}% | On-Time Prob: {ontime_p*100:.1f}% | Risk Level: {risk}")
        print("-" * 75)

    spark.stop()


def main():
    parser = argparse.ArgumentParser(description="Predict Flight Delays (>15 min) with PySpark MLlib")
    parser.add_argument("--carrier", type=str, default=None, help="Operating Carrier (e.g. DL, AA, UA, WN)")
    parser.add_argument("--origin", type=str, default=None, help="Origin airport code (e.g. ORD, ATL, JFK)")
    parser.add_argument("--dest", type=str, default=None, help="Destination airport code (e.g. LAX, SFO, MIA)")
    parser.add_argument("--time", type=int, default=None, help="Scheduled departure HHMM (e.g. 1830)")
    parser.add_argument("--month", type=int, default=10, help="Month of flight (1-12)")
    parser.add_argument("--day", type=int, default=15, help="Day of month (1-31)")
    parser.add_argument("--dow", type=int, default=4, help="Day of week (1=Sun .. 7=Sat)")
    parser.add_argument("--distance", type=float, default=1000.0, help="Distance in miles")
    parser.add_argument("--demo", action="store_true", help="Run multi-case interactive demo")

    args = parser.parse_args()

    if args.carrier and args.origin and args.dest and args.time:
        spark = get_spark_session("FlightDelayCLIInference")
        model = load_model(MODEL_PATH)
        df = prepare_flight_features(
            spark,
            carrier=args.carrier.upper(),
            origin=args.origin.upper(),
            dest=args.dest.upper(),
            dep_time_hhmm=args.time,
            month=args.month,
            day_of_month=args.day,
            day_of_week=args.dow,
            distance=args.distance
        )
        pred, delay_p, ontime_p = predict_flight(model, df)
        status = "DELAYED (>15m)" if pred == 1 else "ON-TIME"
        print(f"\nResult: Flight {args.carrier} {args.origin}->{args.dest} at {args.time:04d}hrs is predicted: {status}")
        print(f"Delay Probability: {delay_p*100:.2f}% | On-time Probability: {ontime_p*100:.2f}%\n")
        spark.stop()
    else:
        run_demo_predictions()


if __name__ == "__main__":
    main()
