"""
stream_producer.py
==================
Simulates real-time live flight operations by writing micro-batches of flight
records into the streaming input folder (data/streaming_input/).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import time
import json
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("StreamProducer")

STREAMING_INPUT_DIR = "data/streaming_input"


def produce_streaming_batches(num_batches: int = 3, records_per_batch: int = 15, delay_between_sec: float = 3.0):
    """
    Generates and saves JSON micro-batches into STREAMING_INPUT_DIR.
    """
    os.makedirs(STREAMING_INPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(int(time.time()))

    carriers = ["DL", "AA", "UA", "WN", "B6", "AS"]
    airports = ["ATL", "ORD", "DFW", "DEN", "LAX", "JFK", "SFO", "SEA", "MCO", "EWR"]

    logger.info("Starting live stream simulation: %d batches of %d records to %s ...",
                num_batches, records_per_batch, STREAMING_INPUT_DIR)

    for b in range(1, num_batches + 1):
        batch_records = []
        for i in range(records_per_batch):
            c = str(rng.choice(carriers))
            orig, dest = rng.choice(airports, size=2, replace=False)
            dep_hour = int(rng.integers(6, 23))
            dep_min = int(rng.choice([0, 15, 30, 45]))
            dep_time = dep_hour * 100 + dep_min
            dist = float(rng.integers(300, 2600))
            crs_elapsed = float(round(45.0 + (dist / 460.0) * 60.0))
            month = int(rng.integers(1, 13))
            day_of_month = int(rng.integers(1, 29))
            day_of_week = int(rng.integers(1, 8))

            # Feature logic matching training pipeline
            tod = "Morning" if 5 <= dep_hour < 12 else ("Afternoon" if 12 <= dep_hour < 17 else ("Evening" if 17 <= dep_hour < 21 else "Night"))
            season = "Winter" if month in [12, 1, 2] else ("Spring" if month in [3, 4, 5] else ("Summer" if month in [6, 7, 8] else "Fall"))
            dist_grp = "ShortHaul" if dist < 500.0 else ("MediumHaul" if dist <= 1500.0 else "LongHaul")

            record = {
                "flight_id": f"FL-{b}-{i+1:03d}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "is_delayed": 0.0,
                "OP_CARRIER": c,
                "ORIGIN": str(orig),
                "DEST": str(dest),
                "DEP_TIME_OF_DAY": tod,
                "SEASON": season,
                "DISTANCE_GROUP": dist_grp,
                "CRS_DEP_HOUR": dep_hour,
                "DAY_OF_WEEK": day_of_week,
                "DAY_OF_MONTH": day_of_month,
                "MONTH": month,
                "DISTANCE": dist,
                "CRS_ELAPSED_TIME": crs_elapsed,
                "CARRIER_DELAY_RATE": float(rng.uniform(0.25, 0.45)),
                "ORIGIN_HOURLY_TRAFFIC": float(rng.integers(10, 50)),
                "DEST_HOURLY_TRAFFIC": float(rng.integers(10, 45)),
                "ORIGIN_DAILY_CUMULATIVE_FLIGHTS": float(rng.integers(15, 180)),
                "class_weight": 1.0
            }
            batch_records.append(record)

        batch_filename = os.path.join(STREAMING_INPUT_DIR, f"batch_{int(time.time())}_{b}.json")
        with open(batch_filename, "w", encoding="utf-8") as f:
            for rec in batch_records:
                f.write(json.dumps(rec) + "\n")

        logger.info("Emitted Batch %d/%d (%d flights) -> %s", b, num_batches, len(batch_records), batch_filename)
        if b < num_batches:
            time.sleep(delay_between_sec)

    logger.info("Live stream simulation completed successfully.")


if __name__ == "__main__":
    produce_streaming_batches()
