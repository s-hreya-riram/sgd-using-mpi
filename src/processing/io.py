import numpy as np
import pandas as pd
from processing.mpi_utils import *
from processing.split import split_test_train
from utils.constants import FEATURE_COLUMNS, LABEL_COLUMN


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
    )

    # Reading each chunk and then putting together the local X, y arrays
    # using the parts accummulated in the previous preprocessing of chunks
    for chunk in reader:
        X, y = chunk[FEATURE_COLUMNS].to_numpy(), chunk[LABEL_COLUMN].to_numpy()
        if X.size > 0:
            X_parts.append(X)
            y_parts.append(y)

    if X_parts:
        X_local = np.vstack(X_parts)
        y_local = np.concatenate(y_parts)
        X_train, y_train, X_test, y_test, train_size, test_size = split_test_train(X_local, y_local, test_ratio=0.3, random_state=42)
        print(f"[Rank {rank}] has split test-train data with {train_size} - {test_size} split")

    
    # Although not the case with the current dataset as it has only 3% missing values,
    # this else case is to handle the edge case of a process that gets no data after preprocessing
    else:
        X_train, X_test = np.empty((0, 0)), np.empty((0, 0))
        y_train, y_test = np.empty((0,)), np.empty((0,))
        print(f"[Rank {rank}] has no data after preprocessing")

    return X_train, y_train, X_test, y_test

def count_rows(file):
    '''
    This function counts the number of rows in the input CSV file.
    This is done so that process with rank 0 can identify the row count
    and broadcast it to the other processes
    '''
    with open(file, "r") as f:
        return sum(1 for _ in f) - 1  # subtract header