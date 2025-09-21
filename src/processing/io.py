import numpy as np
import pandas as pd
from processing.mpi_utils import *
from utils.constants import FEATURE_COLUMNS, LABEL_COLUMN
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self, file, test_ratio, seed, debug, chunksize):
        self.file = file
        self.test_ratio = test_ratio
        self.seed = seed
        self.debug = debug
        self.chunksize = chunksize

        if self.debug:
            logger.setLevel(logging.DEBUG)

    def load_and_split(self):
        return self._read_data()
        
    def _read_data(self):
        """
        Each rank reads its assigned slice of rows from the CSV in chunks.
        Splits data into test and train sets after preprocessing.
        Returns concatenated X_train, y_train, X_test, y_test for that rank.
        """
        logger.info(f"[Rank {rank}] starting to read data from {self.file}")
        
        # Only process with rank 0 counts rows
        if rank == 0:
            n_rows = DatasetLoader.count_rows(self.file)
        else:
            n_rows = None

        # Broadcast row count
        num_rows_total = comm.bcast(n_rows, root=0)

        # Partition rows across ranks almost equally
        rows_per_rank = int(np.ceil(num_rows_total / size))
        begin_index_local = rank * rows_per_rank
        end_index_local = (rank + 1) * rows_per_rank if rank < size - 1 else num_rows_total
        num_rows_local = end_index_local - begin_index_local
        logger.info(f"[Rank {rank}] got {num_rows_local} rows (rows {begin_index_local}–{end_index_local-1})")

        # Skip rows before this rank’s slice
        skip = range(1, begin_index_local + 1) if rank > 0 else None

        # Lists for accumulating split chunks
        X_train_chunks, y_train_chunks, X_test_chunks, y_test_chunks = [], [], [], []
        total_train, total_test = 0, 0

        reader = pd.read_csv(
            self.file,
            header=0,
            skiprows=skip,
            nrows=num_rows_local,
            chunksize=self.chunksize,
            low_memory=True,
        )

        rows_read = 0

        for chunk in reader:

            rows_read += len(chunk)

            X = chunk[FEATURE_COLUMNS].to_numpy()
            y = chunk[LABEL_COLUMN].to_numpy()

            if X.size > 0:
                X_train_chunk, y_train_chunk, X_test_chunk, y_test_chunk, train_size_chunk, test_size_chunk = \
                    DatasetLoader.split_test_train(X, y, self.test_ratio, self.seed)

                if X_train_chunk.size > 0:
                    X_train_chunks.append(X_train_chunk)
                    y_train_chunks.append(y_train_chunk)

                if X_test_chunk.size > 0:
                    X_test_chunks.append(X_test_chunk)
                    y_test_chunks.append(y_test_chunk)

                total_train += train_size_chunk
                total_test += test_size_chunk

        if rows_read == 0:
            logger.warning(f"[Rank {rank}] read 0 rows despite being assigned {num_rows_local} rows from {begin_index_local} to {end_index_local-1}")

        # Concatenate all splits from chunks
        if X_train_chunks and X_test_chunks:
            X_local = np.vstack(X_train_chunks)
            y_local = np.concatenate(y_train_chunks)
            X_test_local = np.vstack(X_test_chunks)
            y_test_local = np.concatenate(y_test_chunks)

            logger.info(f"[Rank {rank}] final split: {X_local.shape[0]} train, {X_test_local.shape[0]} test (from {total_train}+{total_test})")
        else:
            num_features = len(FEATURE_COLUMNS)
            X_local, y_local = np.empty((0, num_features)), np.empty((0,))
            X_test_local, y_test_local = np.empty((0, num_features)), np.empty((0,))

            logger.info(f"[Rank {rank}] has no data after preprocessing")

        logger.info(f"[Rank {rank}] finished reading and splitting data")
        return X_local, y_local, X_test_local, y_test_local

    @staticmethod
    def count_rows(file):
        """Count number of rows in the CSV (minus header)."""
        logger.debug(f"Rank 0 counting rows in {file}")
        with open(file, "r") as f:
            return sum(1 for _ in f) - 1

    @staticmethod
    def split_test_train(X, y, test_ratio, random_seed):
        """Split the data into test and train sets given a test ratio and random seed."""
        logger.debug(f"Rank {rank} splitting data with test ratio {test_ratio}")
        rng = np.random.default_rng(random_seed)

        num_samples = X.shape[0]

        if num_samples == 0:
            return np.empty((0, X.shape[1])), np.empty((0,)), np.empty((0, X.shape[1])), np.empty((0,)), 0, 0
 
        if num_samples == 1: 
            if test_ratio >= 0.5:
                return (np.empty((0, X.shape[1])), np.empty((0,)), X.copy(), y.copy(), 0, 1)
            else:
                return (X.copy(), y.copy(), np.empty((0, X.shape[1])), np.empty((0,)), 1, 0)

 
        indices = rng.permutation(num_samples)

        test_size = int(np.round(num_samples * test_ratio))
        train_size = num_samples - test_size

        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        return X_train, y_train, X_test, y_test, train_size, test_size