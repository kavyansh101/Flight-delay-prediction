"""
data_generator.py
=================
Generates a realistic synthetic flight dataset with 550,000+ records matching
the official US Department of Transportation (DOT) / Kaggle Airline On-Time
Performance schema and statistical distributions.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DataGenerator")

# Airport coordinates (Lat, Lon) for realistic distance calculation
AIRPORT_COORDS = {
    "ATL": (33.6407, -84.4277), "ORD": (41.9742, -87.9073), "DFW": (32.8998, -97.0403),
    "DEN": (39.8561, -104.6737), "CLT": (35.2144, -80.9473), "LAX": (33.9416, -118.4085),
    "JFK": (40.6413, -73.7781), "SFO": (37.6213, -122.3790), "SEA": (47.4502, -122.3088),
    "LAS": (36.0840, -115.1537), "MCO": (28.4312, -81.3081), "EWR": (40.6895, -74.1745),
    "LGA": (40.7769, -73.8740), "BOS": (42.3656, -71.0096), "DTW": (42.2162, -83.3554),
    "MSP": (44.8848, -93.2223), "PHX": (33.4373, -112.0078), "PHL": (39.8729, -75.2437),
    "IAH": (29.9902, -95.3368), "MIA": (25.7959, -80.2870), "BWI": (39.1754, -76.6682),
    "SLC": (40.7899, -111.9791), "SAN": (32.7338, -117.1933), "DCA": (38.8512, -77.0402),
    "TPA": (27.9755, -82.5332)
}

AIRPORTS = list(AIRPORT_COORDS.keys())

# Airport weights reflecting passenger traffic
AIRPORT_WEIGHTS = np.array([
    0.08, 0.07, 0.07, 0.06, 0.05, 0.06, 0.05, 0.04, 0.04, 0.04,
    0.04, 0.04, 0.04, 0.03, 0.03, 0.03, 0.04, 0.03, 0.04, 0.03,
    0.03, 0.03, 0.02, 0.03, 0.02
])
AIRPORT_WEIGHTS /= AIRPORT_WEIGHTS.sum()

CARRIERS = ["WN", "DL", "AA", "UA", "B6", "AS", "NK", "F9", "G4", "HA"]
CARRIER_WEIGHTS = np.array([0.22, 0.18, 0.18, 0.16, 0.07, 0.06, 0.05, 0.04, 0.02, 0.02])
CARRIER_WEIGHTS /= CARRIER_WEIGHTS.sum()

# Carrier delay vulnerability bias (additive minutes)
CARRIER_DELAY_BIAS = {
    "WN": 2.0, "DL": -2.5, "AA": 1.5, "UA": 0.5, "B6": 4.5,
    "AS": -1.5, "NK": 5.0, "F9": 4.0, "G4": 2.5, "HA": -3.0
}

# Airport congestion bias (additive minutes)
AIRPORT_CONGESTION_BIAS = {
    "ORD": 4.0, "EWR": 5.0, "JFK": 4.5, "LGA": 3.5, "SFO": 3.0,
    "ATL": 2.0, "DFW": 1.5, "DEN": 1.5, "LAX": 2.0, "BOS": 2.5
}


def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """Calculates approximate great-circle distance in miles between coordinates."""
    r = 3958.8  # Earth radius in miles
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return np.round(r * c, 1)


def generate_flight_data(num_rows: int = 550000, seed: int = 42, output_csv: str = "data/raw/flights_raw.csv"):
    """
    Generates realistic flight data with 550,000+ rows matching US DOT schema.
    """
    start_time = time.time()
    logger.info("Starting generation of %d synthetic flight records (seed=%d)...", num_rows, seed)
    rng = np.random.default_rng(seed)

    # 1. Dates spanning full year 2024
    base_date = datetime(2024, 1, 1)
    day_offsets = rng.integers(0, 366, size=num_rows)
    dates = [base_date + timedelta(days=int(d)) for d in day_offsets]
    fl_date_strs = np.array([d.strftime("%Y-%m-%d") for d in dates])
    months = np.array([d.month for d in dates])
    days_of_week = np.array([d.isoweekday() for d in dates])  # 1=Mon, 7=Sun

    # 2. Carriers and Flight Numbers
    carriers = rng.choice(CARRIERS, size=num_rows, p=CARRIER_WEIGHTS)
    flight_nums = rng.integers(100, 7999, size=num_rows)

    # 3. Origin and Destination (ensure ORIGIN != DEST)
    origin_indices = rng.choice(len(AIRPORTS), size=num_rows, p=AIRPORT_WEIGHTS)
    dest_shift = rng.integers(1, len(AIRPORTS), size=num_rows)
    dest_indices = (origin_indices + dest_shift) % len(AIRPORTS)

    origins = np.array([AIRPORTS[i] for i in origin_indices])
    dests = np.array([AIRPORTS[i] for i in dest_indices])

    # 4. Distance
    orig_lats = np.array([AIRPORT_COORDS[a][0] for a in origins])
    orig_lons = np.array([AIRPORT_COORDS[a][1] for a in origins])
    dest_lats = np.array([AIRPORT_COORDS[a][0] for a in dests])
    dest_lons = np.array([AIRPORT_COORDS[a][1] for a in dests])
    distances = haversine_distance_vectorized(orig_lats, orig_lons, dest_lats, dest_lons)
    # Ensure minimum practical distance
    distances = np.maximum(distances, 150.0)

    # 5. Scheduled Departure Time (CRS_DEP_TIME: HHMM)
    # Hourly distribution: busy between 06:00 and 21:00
    dep_hours = rng.choice(
        np.arange(5, 24),
        size=num_rows,
        p=np.array([
            0.04, 0.07, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06,
            0.06, 0.06, 0.06, 0.06, 0.07, 0.06, 0.05, 0.03,
            0.01, 0.005, 0.005
        ])
    )
    dep_minutes = rng.choice(np.arange(0, 60, 5), size=num_rows)
    crs_dep_time = dep_hours * 100 + dep_minutes

    # 6. Scheduled Elapsed Time (minutes): roughly ~50 min taxi/climb + (distance / 450 mph * 60)
    crs_elapsed_time = np.round(45.0 + (distances / 460.0) * 60.0).astype(int)

    # Scheduled Arrival Time (CRS_ARR_TIME: HHMM)
    dep_total_mins = dep_hours * 60 + dep_minutes
    arr_total_mins = (dep_total_mins + crs_elapsed_time) % (24 * 60)
    crs_arr_time = (arr_total_mins // 60) * 100 + (arr_total_mins % 60)

    # 7. Cancellations and Diversions (~1.5% cancelled, ~0.2% diverted)
    cancelled = (rng.random(num_rows) < 0.015).astype(int)
    diverted = np.where(cancelled == 1, 0, (rng.random(num_rows) < 0.002).astype(int))

    # Cancellation codes: 'A' (Carrier), 'B' (Weather), 'C' (NAS), 'D' (Security)
    cancel_codes = np.where(
        cancelled == 1,
        rng.choice(["A", "B", "C", "D"], size=num_rows, p=[0.40, 0.45, 0.13, 0.02]),
        ""
    )

    # 8. Delay Factors & Realistic Modeling
    # A) Time-of-day compounding effect: morning flights low delays, night flights high delays
    tod_delay_effect = (dep_hours - 5) * 0.9  # 0 at 5am, +16 mins at 23:00

    # B) Seasonal weather effect: winter spikes (Jan/Feb/Dec) and summer convective storms (Jun/Jul/Aug)
    season_effect = np.where(np.isin(months, [1, 2, 12]), 4.0, 0.0) + \
                    np.where(np.isin(months, [6, 7, 8]), 3.5, 0.0)

    # C) Airport congestion effect
    orig_cong = np.array([AIRPORT_CONGESTION_BIAS.get(a, 0.0) for a in origins])
    dest_cong = np.array([AIRPORT_CONGESTION_BIAS.get(a, 0.0) for a in dests])

    # D) Carrier effect
    carr_bias = np.array([CARRIER_DELAY_BIAS.get(c, 0.0) for c in carriers])

    # Base noise: heavy right-tailed (exponential / gamma mix)
    noise_normal = rng.normal(-5.0, 8.0, size=num_rows)
    has_incident = rng.random(num_rows) < 0.28
    incident_delay = rng.exponential(scale=35.0, size=num_rows) * has_incident

    dep_delay = np.round(
        noise_normal + incident_delay + tod_delay_effect + season_effect + orig_cong + carr_bias
    )
    # Clip extreme values realistically
    dep_delay = np.clip(dep_delay, -25.0, 360.0)

    # Actual elapsed time varies slightly around scheduled elapsed time
    elapsed_variation = rng.normal(1.0, 6.0, size=num_rows)
    actual_elapsed_time = np.maximum(crs_elapsed_time + elapsed_variation, 30.0).astype(int)
    air_time = np.maximum(actual_elapsed_time - rng.integers(20, 38, size=num_rows), 15.0).astype(int)

    # True arrival delay formula
    arr_delay = np.round(dep_delay + (actual_elapsed_time - crs_elapsed_time))
    arr_delay = np.clip(arr_delay, -45.0, 400.0)

    # 9. Actual Departure and Arrival Times (HHMM)
    act_dep_total_mins = (dep_total_mins + dep_delay.astype(int)) % (24 * 60)
    dep_time = (act_dep_total_mins // 60) * 100 + (act_dep_total_mins % 60)

    act_arr_total_mins = (act_dep_total_mins + actual_elapsed_time) % (24 * 60)
    arr_time = (act_arr_total_mins // 60) * 100 + (act_arr_total_mins % 60)

    # 10. Null out actual operational values for cancelled/diverted flights
    dep_time = np.where(cancelled == 1, np.nan, dep_time)
    dep_delay = np.where(cancelled == 1, np.nan, dep_delay)
    arr_time = np.where((cancelled == 1) | (diverted == 1), np.nan, arr_time)
    arr_delay = np.where((cancelled == 1) | (diverted == 1), np.nan, arr_delay)
    actual_elapsed_time = np.where((cancelled == 1) | (diverted == 1), np.nan, actual_elapsed_time)
    air_time = np.where((cancelled == 1) | (diverted == 1), np.nan, air_time)

    # 11. Delay attribution breakdown when ARR_DELAY > 15
    delayed_mask = (arr_delay > 15.0) & (~np.isnan(arr_delay))
    carrier_delay = np.where(delayed_mask, np.round(arr_delay * rng.uniform(0.15, 0.40, size=num_rows)), 0.0)
    weather_delay = np.where(delayed_mask, np.round(arr_delay * rng.uniform(0.10, 0.35, size=num_rows)), 0.0)
    nas_delay = np.where(delayed_mask, np.round(arr_delay * rng.uniform(0.10, 0.30, size=num_rows)), 0.0)
    security_delay = np.where(delayed_mask & (rng.random(num_rows) < 0.02), np.round(arr_delay * 0.05), 0.0)
    late_aircraft_delay = np.where(delayed_mask, np.maximum(0.0, arr_delay - (carrier_delay + weather_delay + nas_delay + security_delay)), 0.0)

    # Assemble DataFrame
    df = pd.DataFrame({
        "FL_DATE": fl_date_strs,
        "OP_CARRIER": carriers,
        "OP_CARRIER_FL_NUM": flight_nums,
        "ORIGIN": origins,
        "DEST": dests,
        "CRS_DEP_TIME": crs_dep_time,
        "DEP_TIME": dep_time,
        "DEP_DELAY": dep_delay,
        "CRS_ARR_TIME": crs_arr_time,
        "ARR_TIME": arr_time,
        "ARR_DELAY": arr_delay,
        "CANCELLED": cancelled,
        "CANCELLATION_CODE": cancel_codes,
        "DIVERTED": diverted,
        "CRS_ELAPSED_TIME": crs_elapsed_time,
        "ACTUAL_ELAPSED_TIME": actual_elapsed_time,
        "AIR_TIME": air_time,
        "DISTANCE": distances,
        "CARRIER_DELAY": carrier_delay,
        "WEATHER_DELAY": weather_delay,
        "NAS_DELAY": nas_delay,
        "SECURITY_DELAY": security_delay,
        "LATE_AIRCRAFT_DELAY": late_aircraft_delay
    })

    # Ensure parent output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    logger.info("Writing %d rows to CSV: %s ...", len(df), output_csv)
    df.to_csv(output_csv, index=False)

    valid_arr_delays = df[df["CANCELLED"] == 0]["ARR_DELAY"].dropna()
    delayed_pct = (valid_arr_delays > 15.0).mean() * 100.0

    elapsed = time.time() - start_time
    logger.info("Successfully generated %d flight records in %.2f seconds.", len(df), elapsed)
    logger.info("Dataset statistics: Cancelled: %.2f%%, Diverted: %.2f%%, Delayed (>15 min): %.2f%%",
                (df["CANCELLED"] == 1).mean() * 100, (df["DIVERTED"] == 1).mean() * 100, delayed_pct)
    return output_csv


if __name__ == "__main__":
    generate_flight_data(num_rows=550000, seed=42, output_csv="data/raw/flights_raw.csv")
