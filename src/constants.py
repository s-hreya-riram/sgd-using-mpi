# Columns after one-hot encoding RatecodeID and payment_type
EXPECTED_RATECODE_COLS = [f"RatecodeID_{i}" for i in [1, 2, 3, 4, 5, 6, 99]]
EXPECTED_PAYMENT_COLS = [f"payment_type_{i}" for i in [1, 2, 3, 4, 5]]

# Columns associated with datetime features
EXPECTED_DATETIME_FEATURES = [
    f"tpep_pickup_datetime_{unit}" for unit in ["day", "month", "year", "hour", "minute", "second"]
] + [
    f"tpep_dropoff_datetime_{unit}" for unit in ["day", "month", "year", "hour", "minute", "second"]
]

# Complete list of features (excluding label)
EXPECTED_SCHEMA = (
    ["passenger_count", "trip_distance", "extra", "PULocationID", "DOLocationID"] 
    + EXPECTED_RATECODE_COLS
    + EXPECTED_PAYMENT_COLS
    + EXPECTED_DATETIME_FEATURES
    + ["trip_duration"]
)