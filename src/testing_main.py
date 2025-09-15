import pandas as pd
import numpy as np
import warnings
from processing.mpi_utils import *
from neural_net import *
from processing.io import read_data
from processing.normalize import normalize

# Suppressing the timestamp parsing warning from pandas to keep the terminal logs clean
warnings.filterwarnings(
    "ignore",
    message="Could not infer format, so each element will be parsed individually"
)
#ignore dtype warning for mixed types in columns inside the chunks to keep the terminal logs clean
warnings.simplefilter(action="ignore", category=pd.errors.DtypeWarning)
    

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, feature_columns, skip_normalization_columns = read_data(
        "../data/nytaxi2022_100rows.csv", header=0, chunksize=10
    )

    print(f"[Rank {rank}] got {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples, {X_train.shape[1]} features.")

    # Debug before normalization
    if X_train.shape[0] > 0:
        print(f"[DEBUG Rank {rank}] Training features BEFORE normalization (first 3 rows):\n{X_train[:3]}")
        print(f"[DEBUG Rank {rank}] Training labels BEFORE normalization:\n{y_train[:10]}")
        print(f"[DEBUG Rank {rank}] Feature columns: {feature_columns}")
        print(f"[DEBUG Rank {rank}] Skip-normalization columns: {skip_normalization_columns}")

    X_train_normalized, y_train_normalized = normalize(X_train, y_train, feature_columns, skip_normalization_columns)

    # Debug after normalization
    if X_train_normalized.shape[0] > 0:
        print(f"[DEBUG Rank {rank}] Training features AFTER normalization (first 3 rows):\n{X_train_normalized[:3]}")
        print(f"[DEBUG Rank {rank}] Training labels AFTER normalization:\n{y_train_normalized[:10]}")

    print(f"[Rank {rank}] got {X_train_normalized.shape[0]} training samples, {X_train_normalized.shape[1]} features.")

    X_test_normalized, y_test_normalized = normalize(X_test, y_test, feature_columns, skip_normalization_columns)

    if X_test_normalized.shape[0] > 0:
        print(f"[DEBUG Rank {rank}] Test features AFTER normalization (first 3 rows):\n{X_test_normalized[:3]}")
        print(f"[DEBUG Rank {rank}] Test labels AFTER normalization:\n{y_test_normalized[:10]}")

    print(f"[Rank {rank}] got {X_test_normalized.shape[0]} testing samples, {X_train_normalized.shape[1]} features.")

    comm.Barrier()
    if rank == 0:
        print("Data distribution and normalization done, ready for SGD...")
   

