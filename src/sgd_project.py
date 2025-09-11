from mpi4py import MPI
import pandas as pd
import numpy as np
import warnings
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
    Returns concatenated X_local, y_local arrays for that rank.
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
        X, y = preprocess_chunk(chunk)
        if X.size > 0:
            X_parts.append(X)
            y_parts.append(y)

    if X_parts:
        X_local = np.vstack(X_parts)
        y_local = np.concatenate(y_parts)
    else:
        X_local = np.empty((0, 0))
        y_local = np.empty((0,))

    return X_local, y_local

def count_rows(file):
    '''
    This function counts the number of rows in the input CSV file.
    This is done so process with rank 0 can identify the row count
    and broadcast it to the other processes
    '''
    with open(file, "r") as f:
        return sum(1 for _ in f) - 1  # subtract header


def preprocess_chunk(df):
    """
    Preprocess one chunk of the taxi dataset.
    - narrowing down to the columns mentioned in the problem statement
    - drop NAs and filter invalid data
    - TODO
        - add some form of encoding for categorical features?
        - normalize data
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

    X = df.drop(columns=["total_amount"]).values
    y = df["total_amount"].values
    return X, y

def get_datetime_features(df, col_name):
    '''
    Derive datetime features from a datetime column
    '''
    dt = df[col_name].dt
    features = pd.DataFrame({
        col_name + "_day": dt.day,
        col_name + "_month": dt.month,
        col_name + "_year": dt.year,
        col_name + "_hour": dt.hour,
        col_name + "_minute": dt.minute,
        col_name + "_second": dt.second,
    }, index=df.index)
    return pd.concat([df, features], axis=1)

# ----------------------------------------------------
# Main
# ----------------------------------------------------
if __name__ == "__main__":
    # It takes a few mins to read and process data from the original file 
    # but the CPU utilization is mostly within 75% on my machine
    # for implementation & testing purposes, using the subset of 1MM rows for now
    #X_local, y_local = read_data("../data/nytaxi2022.csv", header=0, chunksize=100000)
    X_local, y_local = read_data("../data/nytaxi2022_subset.csv", header=0, chunksize=100000)    
    print(f"[Rank {rank}] got {X_local.shape[0]} samples, {X_local.shape[1]} features.")
    comm.Barrier()
    if rank == 0:
        print("Data distribution done, ready for normalization/SGD...")
