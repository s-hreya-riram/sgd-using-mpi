from utils import get_datetime_features
from constants import EXPECTED_SCHEMA
import numpy as np
import pandas as pd

def preprocess_chunk(df):
    """
    Preprocess one chunk of the taxi dataset.
    - narrowing down to the columns mentioned in the problem statement
    - drop NAs and filter invalid data
    - add encoding for categorical variables (one-hot or frequency encoding depending on frequency)
    Returns (X, y).
    """
    df = df[
        [
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
    ].copy()
    
    # Drop rows with any NA values
    df.dropna(inplace=True)
    
    # Clean extra and total_amount values to be non-negative and positive respectively 
    # based on the attribute descriptions from Kaggle: https://www.kaggle.com/datasets/diishasiing/revenue-for-cab-drivers/data
    df = df[df["extra"] >= 0]
    df = df[df["total_amount"] > 0]

    # Convert datetime columns (vectorized)
    df.loc[:, "tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df.loc[:, "tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")

    # Derive datetime features
    df = get_datetime_features(df, "tpep_pickup_datetime")
    df = get_datetime_features(df, "tpep_dropoff_datetime")

    # Trip duration in minutes
    df["trip_duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60

    # Drop original datetime cols
    df.drop(columns=["tpep_pickup_datetime", "tpep_dropoff_datetime"], inplace=True)

    # Since there are 6 unique values for RatecodeID, 263 for PULocationID, 262 for DOLocationID and 5 for payment_type
    # taking into account the volume of data, using one hot encoding for ratecodeId and payment_type,
    # using frequency encoding for PULocationID and DOLocationID

    # One-hot encoding RatecodeID and payment_type, having to manually specify expected columns to address the issue from 
    # testing - some chunks may not have all categories, that was causing inconsistent number of columns
    # across chunks and causing errors like "ValueError: all the input array dimensions except for the concatenation axis
    # must match exactly, but along dimension 1, the array at index 0 has size 28 and the array at index 1 has size 29"
    # hardcoded values are from the exploratory data analysis notebook
    # TODO explore refactoring this to avoid hardcoding
    expected_ratecode_cols = [f"RatecodeID_{i}" for i in [1, 2, 3, 4, 5, 6, 99]]
    expected_payment_cols = [f"payment_type_{i}" for i in [1, 2, 3, 4, 5]]

    df = pd.get_dummies(df, columns=["RatecodeID", "payment_type"], prefix=["RatecodeID", "payment_type"])

    for col in ["PULocationID", "DOLocationID"]:
        freq = df[col].value_counts(normalize=True)
        df[col] = df[col].map(freq)

    # Add missing dummy columns with 0s
    for col in expected_ratecode_cols + expected_payment_cols:
        if col not in df:
            df[col] = 0

    # Keep column order consistent
    df = df.reindex(columns=EXPECTED_SCHEMA + ["total_amount"], fill_value=0)

    # Tracking feature_columns and skip_normalization_columns to skip normalization of the attributes
    # that are the derived date-time attributes, were one-hot encoded or frequency encoded above
    # TODO see if there is an alternative to manually specifying the column names
    skip_normalization_columns = [
        col for col in df.columns
        if col.startswith("RatecodeID_")
        or col.startswith("payment_type_")
        or col.startswith("tpep_pickup_datetime_")
        or col.startswith("tpep_dropoff_datetime_")
        or col in ["PULocationID", "DOLocationID"]
    ]
    feature_columns = [c for c in df.columns if c != "total_amount"]

    # ensuring X and y are of type float64 as object type arrays cause errors with MPI Allreduce
    X = df[feature_columns].values.astype(np.float64) 
    y = df["total_amount"].values.astype(np.float64)

    return X, y, feature_columns, skip_normalization_columns
