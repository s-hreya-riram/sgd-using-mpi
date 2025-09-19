import numpy as np
from utils.activation_functions import *

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
FEATURE_COLUMNS = (
    ["passenger_count", "trip_distance", "extra", "PULocationID", "DOLocationID"]
    + EXPECTED_RATECODE_COLUMNS
    + EXPECTED_PAYMENT_COLUMNS
    + EXPECTED_DATETIME_FEATURES
    + ["trip_duration"]
)

SKIP_NORMALIZATION_COLUMNS = []
# uncomment as necessary to skip normalization of categorical features
#SKIP_NORMALIZATION_COLUMNS = ["PULocationID", "DOLocationID"] + EXPECTED_RATECODE_COLUMNS + EXPECTED_PAYMENT_COLUMNS

LABEL_COLUMN = "total_amount"

ACTIVATION_FUNCTION_MAP = {
    "linear": (linear, linear_grad),
    "relu": (relu, relu_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
}