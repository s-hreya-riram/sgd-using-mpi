import pandas as pd
import numpy as np
import warnings
from processing.mpi_utils import *
from neural_net import *
from processing.io import read_data
from processing.normalize import normalize
from constants import FEATURE_COLUMNS, SKIP_NORMALIZATION_COLUMNS

# Suppressing the timestamp parsing warning from pandas to keep the terminal logs clean
warnings.filterwarnings(
    "ignore",
    message="Could not infer format, so each element will be parsed individually"
)
#ignore dtype warning for mixed types in columns inside the chunks to keep the terminal logs clean
warnings.simplefilter(action="ignore", category=pd.errors.DtypeWarning)

if __name__ == "__main__":

    X_train, y_train, X_test, y_test = read_data("../data/processed/nytaxi2022_preprocessed_complete.csv", header=0, chunksize=100000)
    #X_train, y_train, X_test, y_test, feature_columns, skip_normalization_columns = read_data("../data/nytaxi2022_subset.csv", header=0, chunksize=100000)
    print(f"[Rank {rank}] got {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples, {X_train.shape[1]} features.")
    X_train_normalized, y_train_normalized = normalize(X_train, y_train, FEATURE_COLUMNS, SKIP_NORMALIZATION_COLUMNS)
    print(f"[Rank {rank}] got {X_train_normalized.shape[0]} training samples, {X_train_normalized.shape[1]} features.")
    X_test_normalized, y_test_normalized = normalize(X_test, y_test, FEATURE_COLUMNS, SKIP_NORMALIZATION_COLUMNS)
    print(f"[Rank {rank}] got {X_test_normalized.shape[0]} testing samples, {X_train_normalized.shape[1]} features.")
    comm.Barrier()
    if rank == 0:
        print("Data distribution and normalization done, ready for SGD...")


