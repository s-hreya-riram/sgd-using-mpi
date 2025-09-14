from mpi4py import MPI
import pandas as pd
import numpy as np
import warnings
from preprocess import preprocess_chunk
from utils import count_rows, get_datetime_features, split_test_train

# Suppressing the timestamp parsing warning from pandas to keep the terminal logs clean
warnings.filterwarnings(
    "ignore",
    message="Could not infer format, so each element will be parsed individually"
)
#ignore dtype warning for mixed types in columns inside the chunks to keep the terminal logs clean
warnings.simplefilter(action="ignore", category=pd.errors.DtypeWarning)

# ----------------------------------------------------
# MPI globals
# ----------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ----------------------------------------------------
# Functions
# ----------------------------------------------------
def read_data(file, header=0, chunksize=100000):
    """
    Each rank reads its assigned slice of rows from the CSV in chunks.
    Splits data into test and train sets after preprocessing
    Returns concatenated X_train, y_train, X_test, y_test  for that rank.
    TODO evaluate performance of different chunk sizes with the actual dataset
    TODO change print statements to logging throughout the code
    """
    # Only process with rank 0 counts rows from the file so 
    # individual processes don't have to read the entire file
    if rank == 0:
        n_rows = count_rows(file)
    else:
        n_rows = None

    # Process with rank 0 broadcasts the row count to all processes
    num_rows_total = comm.bcast(n_rows, root=0)

    # Partition rows across ranks almost equally (as num_rows_total may not be divisible by size)
    rows_per_rank = num_rows_total // size
    begin_index_local = rank * rows_per_rank
    end_index_local = (rank + 1) * rows_per_rank if rank < size - 1 else num_rows_total
    num_rows_local = end_index_local - begin_index_local
    print(f"[Rank {rank}] got {num_rows_local} rows to process (rows {begin_index_local} to {end_index_local-1})")

    # Skip rows before this rank’s slice - rank 0 doesn't skip any rows
    skip = range(1, begin_index_local + 1) if rank > 0 else None

    # Accumulate chunks locally
    X_parts, y_parts = [], []

    # reading assigned chunk for each process
    # handling date parsing here itself as this seems more optimal than using
    # pd.to_datetime() later on the entire chunk
    reader = pd.read_csv(
        file,
        header=header,
        skiprows=skip,
        nrows=num_rows_local,
        chunksize=chunksize,
        low_memory=True, 
        parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"]
    )

    # Reading each chunk and then putting together the local X, y arrays
    # using the parts accummulated in the previous preprocessing of chunks
    for chunk in reader:
        X, y, feature_columns, skip_normalization_columns = preprocess_chunk(chunk)
        if X.size > 0:
            X_parts.append(X)
            y_parts.append(y)

    if X_parts:
        X_local = np.vstack(X_parts)
        y_local = np.concatenate(y_parts)
        X_train, y_train, X_test, y_test, train_size, test_size = split_test_train(X_local, y_local, test_ratio=0.2, random_state=42)
        print(f"[Rank {rank}] has split test-train data with {train_size} - {test_size} split")

    
    # Although not the case with the current dataset as it has only 3% missing values,
    # this else case is to handle the edge case of a process that gets no data after preprocessing
    else:
        X_train, X_test = np.empty((0, 0)), np.empty((0, 0))
        y_train, y_test = np.empty((0,)), np.empty((0,))
        print(f"[Rank {rank}] has no data after preprocessing")

    return X_train, y_train, X_test, y_test, feature_columns, skip_normalization_columns

def normalize(training_feature_local, training_label_local, feature_columns, skip_normalization_columns=[]):
    """Following the same process as professor did in Kernel Ridge Regression,
    calculating local sum/sq diff and then using Allreduce to get global sum/sqdiff."""
    
    # calculating number of features 
    n_feature = training_feature_local.shape[1] if training_feature_local.size else 0

    #determine the indices of the columns to be normalized
    skip_indices = [feature_columns.index(col) for col in skip_normalization_columns if col in feature_columns]
    normalize_indices = [i for i in range(n_feature) if i not in skip_indices]

    # feature normalization
    if training_feature_local.size:
        local_feature_sum = np.sum(training_feature_local[:, normalize_indices], axis=0)
    else:
        local_feature_sum = np.zeros(len(normalize_indices))

    global_feature_sum = np.zeros_like(local_feature_sum)
    comm.Allreduce(local_feature_sum, global_feature_sum, op=MPI.SUM)

    local_feature_count = np.array([training_feature_local.shape[0]], dtype=np.int64)
    global_feature_count = np.array([0], dtype=np.int64)
    comm.Allreduce(local_feature_count, global_feature_count, op=MPI.SUM)

    feature_mean = global_feature_sum / global_feature_count[0]

    if training_feature_local.size:
        local_sqdiff = np.sum((training_feature_local[:, normalize_indices] - feature_mean) ** 2, axis=0)
    else:
        local_sqdiff = np.zeros(len(normalize_indices))

    global_sqdiff = np.zeros_like(local_sqdiff)
    comm.Allreduce(local_sqdiff, global_sqdiff, op=MPI.SUM)
    feature_std = np.sqrt(global_sqdiff / global_feature_count[0])

    # normalize in place, only for normalize_indices
    if training_feature_local.size > 0:
        feature_std_nozero = np.where(feature_std == 0, 1.0, feature_std) # covering edge case of stddev being 0
        training_feature_local[:, normalize_indices] = (training_feature_local[:, normalize_indices] - feature_mean) / feature_std_nozero

    # label normalization
    local_label_sum = np.array([np.sum(training_label_local)], dtype=np.float64)
    global_label_sum = np.array([0.0], dtype=np.float64)
    comm.Allreduce(local_label_sum, global_label_sum, op=MPI.SUM)

    local_label_count = np.array([training_label_local.shape[0]], dtype=np.int64)
    global_label_count = np.array([0], dtype=np.int64)
    comm.Allreduce(local_label_count, global_label_count, op=MPI.SUM)

    label_mean = global_label_sum[0] / global_label_count[0]

    local_sqdiff = np.array([np.sum((training_label_local - label_mean) ** 2)], dtype=np.float64)
    global_sqdiff = np.array([0.0], dtype=np.float64)
    comm.Allreduce(local_sqdiff, global_sqdiff, op=MPI.SUM)
    label_std = np.sqrt(global_sqdiff[0] / global_label_count[0])

    if training_label_local.size > 0:
        std_nozero = label_std if label_std > 0 else 1.0
        training_label_local = (training_label_local - label_mean) / std_nozero

    return training_feature_local, training_label_local

# ----------------------------------------------------
# Main
# ----------------------------------------------------
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
