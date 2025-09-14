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

def count_rows(file):
    '''
    This function counts the number of rows in the input CSV file.
    This is done so that process with rank 0 can identify the row count
    and broadcast it to the other processes
    '''
    with open(file, "r") as f:
        return sum(1 for _ in f) - 1  # subtract header


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

    # Add missing dummy columns with 0s
    for col in expected_ratecode_cols + expected_payment_cols:
        if col not in df:
            df[col] = 0

    # Keep column order consistent
    df = df.reindex(columns=sorted(df.columns))

    # frequency encode PULocationID and DOLocationID
    for col in ["PULocationID", "DOLocationID"]:
        freq = df[col].value_counts(normalize=True)
        df[col] = df[col].map(freq)

    # Tracking feature_columns and skip_normalization_columns to skip normalization of the attributes
    # that are the derived date-time attributes, were one-hot encoded or frequency encoded above
    # TODO see if there is an alternative to manually specifying the column names
    skip_normalization_columns = [column for column in df.columns if column.startswith("RatecodeID_") or 
                        column.startswith("payment_type_") or 
                        column.startswith("tpep_pickup_datetime_") or
                        column.startswith("tpep_dropoff_datetime_") or
                        column in ["PULocationID", "DOLocationID"]]
    feature_columns = [c for c in df.columns if c != "total_amount"]
    # ensuring X and y are of type float64 as object type arrays cause errors with MPI Allreduce
    X = df[feature_columns].values.astype(np.float64) 
    y = df["total_amount"].values.astype(np.float64)

    return X, y, feature_columns, skip_normalization_columns

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

def split_test_train(X, y, test_ratio, random_state):
    '''
    Split the data into test and train sets given a test ratio and a random seed
    Returns X_train, y_train, X_test, y_test in that order
    '''
    # setting a random seed to make the shuffling deterministic
    np.random.seed(random_state)

    num_samples = X.shape[0]
    indices = np.arange(num_samples)

    # shuffling indices so the test-train split is random
    np.random.shuffle(indices)

    test_size = int(num_samples * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, y_train, X_test, y_test, (num_samples - test_size), test_size

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
    #X_local, y_local = read_data("../data/nytaxi2022.csv", header=0, chunksize=100000)
    X_train, y_train, X_test, y_test, feature_columns, skip_normalization_columns = read_data("../data/nytaxi2022.csv", header=0, chunksize=100000)    
    print(f"[Rank {rank}] got {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples, {X_train.shape[1]} features.")
    X_train_normalized, y_train_normalized = normalize(X_train, y_train, feature_columns, skip_normalization_columns)
    print(f"[Rank {rank}] got {X_train_normalized.shape[0]} training samples, {X_train_normalized.shape[1]} features.")
    X_test_normalized, y_test_normalized = normalize(X_test, y_test, feature_columns, skip_normalization_columns) # placeholder
    print(f"[Rank {rank}] got {X_test_normalized.shape[0]} training samples, {X_train_normalized.shape[1]} features.")
    comm.Barrier()
    if rank == 0:
        print("Data distribution and normalization done, ready for SGD...")
