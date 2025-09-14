import numpy as np
import pandas as pd

def count_rows(file):
    '''
    This function counts the number of rows in the input CSV file.
    This is done so that process with rank 0 can identify the row count
    and broadcast it to the other processes
    '''
    with open(file, "r") as f:
        return sum(1 for _ in f) - 1  # subtract header
    
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