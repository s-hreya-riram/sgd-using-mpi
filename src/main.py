import os
import sys
import pandas as pd
import numpy as np
import warnings
from processing.mpi_utils import *
from neural_net import *
from processing.io import DatasetLoader
from processing.normalize import Normalizer
from utils.constants import FEATURE_COLUMNS, SKIP_NORMALIZATION_COLUMNS

# Suppressing the timestamp parsing warning from pandas to keep the terminal logs clean
warnings.filterwarnings(
    "ignore",
    message="Could not infer format, so each element will be parsed individually"
)
#ignore dtype warning for mixed types in columns inside the chunks to keep the terminal logs clean
warnings.simplefilter(action="ignore", category=pd.errors.DtypeWarning)

# Validation functions for user inputs
def validateFilePath(file_path):
    if not os.path.isfile(file_path):
        raise ValueError(f"File not found: {file_path}")
    return file_path

def validatePositiveInteger(value):
    int_value = int(value)
    if int_value <= 0:
        raise ValueError("Value must be a positive integer")
    return int_value

def validateFloatDecimals(value):
    float_value = float(value)
    if not (0 < float_value < 1):
        raise ValueError("Value must be between 0 and 1")
    return float_value

def validateActivationFunction(value):
    valid_functions = ["relu", "sigmoid", "tanh", "linear"]
    if value not in valid_functions:
        raise ValueError(f"Activation function must be one of {valid_functions}")
    return value

if __name__ == "__main__":

    # get user input for the following attributes:
    # file path, hidden layer dimensions, learning rate, activation function, epochs, batch size, seed

    if len(sys.argv) != 9:
        print("Usage: python script.py <file_path> <test_ratio> <hidden_dimensions> <learning_rate> <activation_function> <epochs> <batch_size> <seed>")
        sys.exit(1)

    try:
        file_path = validateFilePath(sys.argv[1])
        test_ratio = validateFloatDecimals(sys.argv[2])
        hidden_dim = validatePositiveInteger(sys.argv[3])
        lr = validateFloatDecimals(sys.argv[4])
        activation = validateActivationFunction(sys.argv[5])
        epochs = validatePositiveInteger(sys.argv[6])
        batch_size = validatePositiveInteger(sys.argv[7])
        seed = validatePositiveInteger(sys.argv[8])
    except ValueError as e:
        print(f"Input error: {e}")
        sys.exit(1)

    # Print out user inputs back
    if rank == 0:
        print("User inputs:")
        print(f"File: {file_path}")
        print(f"Test ratio for train-test split: {test_ratio}")
        print(f"Hidden dim: {hidden_dim}")
        print(f"Learning rate: {lr}")
        print(f"Activation: {activation}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Seed: {seed}")

    data_loader = DatasetLoader(file_path, test_ratio, seed, chunksize=100000)
    X_train, y_train, X_test, y_test = data_loader.load_and_split()
    print(f"[Rank {rank}] got {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples, {X_train.shape[1]} features.")


    normalizer = Normalizer(FEATURE_COLUMNS, SKIP_NORMALIZATION_COLUMNS)
    X_train_normalized, y_train_normalized = normalizer.normalize(X_train, y_train)
    print(f"[Rank {rank}] got {X_train_normalized.shape[0]} training samples, {X_train_normalized.shape[1]} features.")
    X_test_normalized, y_test_normalized = normalizer.normalize_test_data(X_test, y_test)
    print(f"[Rank {rank}] got {X_test_normalized.shape[0]} testing samples, {X_test_normalized.shape[1]} features.")
    
    comm.Barrier()
    
    if rank == 0:
        print("Data distribution and normalization done, ready for SGD...")
    
    input_dim = X_train.shape[1]
    model = NeuralNet(input_dim, hidden_dim, lr, activation, seed)
    execute_model(model, X_train_normalized, y_train_normalized, X_test_normalized, y_test_normalized, epochs, batch_size, seed)