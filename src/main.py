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
    # It takes a few mins to read and process data from the original file 
    # but the CPU utilization is mostly within 75% on my machine
    # for implementation & testing purposes, using the subset of 1MM rows for now
    #X_local, y_local, X_test, y_test, feature_columns, skip_normalization_columns = read_data("../data/nytaxi2022.csv", header=0, chunksize=100000)
    X_train, y_train, X_test, y_test, feature_columns, skip_normalization_columns = read_data("../data/nytaxi2022_subset.csv", header=0, chunksize=100000)
    print(f"[Rank {rank}] got {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples, {X_train.shape[1]} features.")
    X_train_normalized, y_train_normalized = normalize(X_train, y_train, feature_columns, skip_normalization_columns)
    print(f"[Rank {rank}] got {X_train_normalized.shape[0]} training samples, {X_train_normalized.shape[1]} features.")
    X_test_normalized, y_test_normalized = normalize(X_test, y_test, feature_columns, skip_normalization_columns) # placeholder
    print(f"[Rank {rank}] got {X_test_normalized.shape[0]} training samples, {X_train_normalized.shape[1]} features.")
    comm.Barrier()
    if rank == 0:
        print("Data distribution and normalization done, ready for SGD...")


