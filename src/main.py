
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import os

# Limiting the number of threads to control CPU utilization
# os.environ["OMP_NUM_THREADS"] = "4"
# os.environ["MKL_NUM_THREADS"] = "4"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import pandas as pd
import numpy as np
import warnings
from processing.mpi_utils import *
from neural_net import *
from processing.io import DatasetLoader
from processing.normalize import Normalizer
from utils.constants import FEATURE_COLUMNS, SKIP_NORMALIZATION_COLUMNS
import logging


# Suppressing the timestamp parsing warning from pandas to keep the terminal logs clean
warnings.filterwarnings(
    "ignore",
    message="Could not infer format, so each element will be parsed individually"
)
#ignore dtype warning for mixed types in columns inside the chunks to keep the terminal logs clean
warnings.simplefilter(action="ignore", category=pd.errors.DtypeWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    valid_functions = ["relu", "sigmoid", "tanh", "leaky_relu", "elu"]
    if value not in valid_functions:
        raise ValueError(f"Activation function must be one of {valid_functions}")
    return value

def validateYesNoInput(value):
    if value not in ['y', 'n']:
        raise ValueError("Input must be 'y' or 'n'")
    return value == 'y'

if __name__ == "__main__":

    # get user input for the following attributes:
    # file path, hidden layer dimensions, learning rate, activation function, max iterations, batch size, seed
    if len(sys.argv) != 10:
        print("Usage: python main.py <file_path> <test_ratio> <hidden_dimensions> <learning_rate> <activation_function> <max_iterations> <batch_size> <seed> <debug (y/n)")
        sys.exit(1)

    try:
        file_path = validateFilePath(sys.argv[1])
        test_ratio = validateFloatDecimals(sys.argv[2])
        hidden_dim = validatePositiveInteger(sys.argv[3])
        learning_rate = validateFloatDecimals(sys.argv[4])
        activation = validateActivationFunction(sys.argv[5])
        max_iterations = validatePositiveInteger(sys.argv[6])
        batch_size = validatePositiveInteger(sys.argv[7])
        seed = validatePositiveInteger(sys.argv[8])
        debug_mode = validateYesNoInput(sys.argv[9].lower())
        if debug_mode:
            logger.setLevel(logging.DEBUG) 

    except ValueError as e:
        logger.error(f"Input error: {e}")
        sys.exit(1)

    # Print out user inputs back
    if rank == 0:
        logger.info("User inputs:")
        logger.info(f"File: {file_path}")
        logger.info(f"Test ratio for train-test split: {test_ratio}")
        logger.info(f"Hidden dim: {hidden_dim}")
        logger.info(f"Base learning rate: {learning_rate}")
        logger.info(f"Activation: {activation}")
        logger.info(f"Max iterations: {max_iterations}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Seed: {seed}")

    dataset_loader = DatasetLoader(file_path, test_ratio, seed, debug=debug_mode, chunksize=100000)
    X_train, y_train, X_test, y_test = dataset_loader.load_and_split()
    logger.info(f"[Rank {rank}] got {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples, {X_train.shape[1]} features.")

    normalizer = Normalizer(FEATURE_COLUMNS, SKIP_NORMALIZATION_COLUMNS, debug=debug_mode)
    X_train_normalized, y_train_normalized = normalizer.normalize(X_train, y_train)
    logger.info(f"[Rank {rank}] got {X_train_normalized.shape[0]} training samples, {X_train_normalized.shape[1]} features.")
    X_test_normalized, y_test_normalized = normalizer.normalize_test_data(X_test, y_test)
    logger.info(f"[Rank {rank}] got {X_test_normalized.shape[0]} testing samples, {X_test_normalized.shape[1]} features.")

    # adding a barrier to ensure all ranks start running the model together
    comm.Barrier()
    
    if rank == 0:
        logger.info("Data distribution and normalization done, ready for SGD...")
    
    input_dim = len(FEATURE_COLUMNS) 
    model = NeuralNet(input_dim, hidden_dim, learning_rate, activation, size, seed, debug=debug_mode)
    execute_model(model, X_train_normalized, y_train_normalized, X_test_normalized, y_test_normalized, max_iterations, batch_size, seed)