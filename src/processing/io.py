import numpy as np
import pandas as pd
from processing.mpi_utils import *
from utils.constants import FEATURE_COLUMNS, LABEL_COLUMN
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self, file, test_ratio, seed, chunksize):
        self.file = file
        self.test_ratio = test_ratio
        self.seed = seed
        self.chunksize = chunksize

    def load_and_split(self):
        return self._read_data()
        
    def _read_data(self):
        """
        Each rank reads its assigned slice of rows from the CSV in chunks.
        Splits data into test and train sets after preprocessing
        Returns concatenated X_train, y_train, X_test, y_test  for that rank.
        """
        logger.debug(f"[Rank {rank}] starting to read data from {self.file}")
        # Only process with rank 0 counts rows from the file so
        # individual processes don't have to read the entire file
        if rank == 0:
            n_rows = DatasetLoader.count_rows(self.file)
        else:
            n_rows = None

        # Process with rank 0 broadcasts the row count to all processes
        num_rows_total = comm.bcast(n_rows, root=0)

        # Partition rows across ranks almost equally
        rows_per_rank = int(np.ceil(num_rows_total / size))
        begin_index_local = rank * rows_per_rank
        end_index_local = (rank + 1) * rows_per_rank if rank < size - 1 else num_rows_total
        num_rows_local = end_index_local - begin_index_local
        # Add a log statement for debugging purposes
        logger.debug(f"[Rank {rank}] got {num_rows_local} rows to process (rows {begin_index_local} to {end_index_local-1})")

        # Skip rows before this rank’s slice - rank 0 doesn't skip any rows
        skip = range(1, begin_index_local + 1) if rank > 0 else None

        # maintain list of parts from chunks to merge later
        X_parts, y_parts = [], []

        # reading assigned chunk for each process
        reader = pd.read_csv(
            self.file,
            header=0,
            skiprows=skip,
            nrows=num_rows_local,
            chunksize=self.chunksize,
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
            X_train, y_train, X_test, y_test, train_size, test_size = DatasetLoader.split_test_train(X_local, y_local, self.test_ratio, self.seed)
            logger.info(f"[Rank {rank}] has split test-train data with {train_size} - {test_size} split")

        
        # Although not the case with the current dataset has no missing values,
        # this else case is to handle the edge case of a process that gets no data
        else:
            num_features = len(FEATURE_COLUMNS)
            X_train, X_test = np.empty((0, num_features)), np.empty((0, num_features))
            y_train, y_test = np.empty((0,)), np.empty((0,))

            logger.info(f"[Rank {rank}] has no data after preprocessing")
        logger.info(f"[Rank {rank}] finished reading and splitting data")
        return X_train, y_train, X_test, y_test

    @staticmethod
    def count_rows(file):
        '''
        This function counts the number of rows in the input CSV file.
        This is done so that process with rank 0 can identify the row count
        and broadcast it to the other processes
        '''
        logger.debug(f"Rank 0 counting rows in {file}")
        with open(file, "r") as f:
            return sum(1 for _ in f) - 1  # subtract header

    @staticmethod
    def split_test_train(X, y, test_ratio, random_seed):
        '''
        Split the data into test and train sets given a test ratio and a random seed
        Returns X_train, y_train, X_test, y_test, train_size, test_size
        '''
        logger.info(f"Rank {rank} splitting data into test and train sets with test ratio {test_ratio}")
        # setting a random seed to make the shuffling deterministic
        np.random.seed(random_seed)

        logger.debug(f"Rank {rank} data has {X.shape[0]} samples and {X.shape[1]} features")
        num_samples = X.shape[0]
        indices = np.arange(num_samples)

        # shuffling indices so the test-train split is random
        logger.debug(f"Rank {rank} shuffling data indices for random test-train split")
        np.random.shuffle(indices)

        test_size = int(num_samples * test_ratio)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        logger.debug(f"Rank {rank} selected {test_size} samples for testing")

        X_train, y_train = X[train_indices], y[train_indices]
        logger.debug(f"Rank {rank} training data has {X_train.shape[0]} samples")
        X_test, y_test = X[test_indices], y[test_indices]
        logger.debug(f"Rank {rank} test data has {X_test.shape[0]} samples")
        logger.debug(f"Rank {rank} completed test-train split: {num_samples - test_size} train samples, {test_size} test samples")
        return X_train, y_train, X_test, y_test, (num_samples - test_size), test_size