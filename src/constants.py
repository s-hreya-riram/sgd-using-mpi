# Define expected input columns
EXPECTED_INPUT_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "extra",
    "total_amount",
]

# Columns after one-hot encoding RatecodeID and payment_type
EXPECTED_RATECODE_COLUMNS = [f"RatecodeID_{i}" for i in [1, 2, 3, 4, 5, 6, 99]]
EXPECTED_PAYMENT_COLUMNS = [f"payment_type_{i}" for i in [1, 2, 3, 4, 5]]


# Columns associated with datetime features
EXPECTED_DATETIME_FEATURES = [
    f"tpep_pickup_datetime_{unit}" for unit in ["day", "month", "year", "hour", "minute", "second"]
] + [
    f"tpep_dropoff_datetime_{unit}" for unit in ["day", "month", "year", "hour", "minute", "second"]
]

# Complete list of features (excluding label)
EXPECTED_SCHEMA = (
    ["passenger_count", "trip_distance", "extra", "PULocationID", "DOLocationID"] 
    + EXPECTED_RATECODE_COLUMNS
    + EXPECTED_PAYMENT_COLUMNS
    + EXPECTED_DATETIME_FEATURES
    + ["trip_duration"]
)

FEATURE_COLUMNS = ['passenger_count', 'trip_distance', 'extra', 'PULocationID', 'DOLocationID', 'RatecodeID_1', 'RatecodeID_2', 'RatecodeID_3', 'RatecodeID_4', 'RatecodeID_5', 'RatecodeID_6', 'RatecodeID_99', 'payment_type_1', 'payment_type_2', 'payment_type_3', 'payment_type_4', 'payment_type_5', 'tpep_pickup_datetime_day', 'tpep_pickup_datetime_month', 'tpep_pickup_datetime_year', 'tpep_pickup_datetime_hour', 'tpep_pickup_datetime_minute', 'tpep_pickup_datetime_second', 'tpep_dropoff_datetime_day', 'tpep_dropoff_datetime_month', 'tpep_dropoff_datetime_year', 'tpep_dropoff_datetime_hour', 'tpep_dropoff_datetime_minute', 'tpep_dropoff_datetime_second', 'trip_duration']
SKIP_NORMALIZATION_COLUMNS = ['PULocationID', 'DOLocationID', 'RatecodeID_1', 'RatecodeID_2', 'RatecodeID_3', 'RatecodeID_4', 'RatecodeID_5', 'RatecodeID_6', 'RatecodeID_99', 'payment_type_1', 'payment_type_2', 'payment_type_3', 'payment_type_4', 'payment_type_5', 'tpep_pickup_datetime_day', 'tpep_pickup_datetime_month', 'tpep_pickup_datetime_year', 'tpep_pickup_datetime_hour', 'tpep_pickup_datetime_minute', 'tpep_pickup_datetime_second', 'tpep_dropoff_datetime_day', 'tpep_dropoff_datetime_month', 'tpep_dropoff_datetime_year', 'tpep_dropoff_datetime_hour', 'tpep_dropoff_datetime_minute', 'tpep_dropoff_datetime_second']
LABEL_COLUMN = "total_amount"