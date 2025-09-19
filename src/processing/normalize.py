import numpy as np
from processing.mpi_utils import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Normalizer:
    """
    Following the same process as Kernel Ridge Regression from Lecture 4,
    calculating local sum/sq diff for each process and then using allreduce to get global sum/sqdiff.

    Additionally this takes in the feature columns and any columns to be skipped from normalization
    The reason for including this logic is to evaluate the test rmse with/without normalizing 
    features that were frequency encoded and(or) are categorical
    """
    def __init__(self, feature_columns, skip_normalization_columns):
        logger.info("Starting Normalizer")
        self.feature_columns = feature_columns
        self.skip_normalization_columns = skip_normalization_columns

        # attributes after normalizing train data that will be used for normalizing test data
        self.x_train_mean = None
        self.x_train_std = None
        self.y_train_mean = None
        self.y_train_std = None

    def normalize(self, X_train_local, y_train_local):
        '''calculate local sum/sq diff for each process and then using allreduce 
        to get global sum/sqdiff to normalize the training feature and label vectors.'''
        logger.info(f"Starting normalization of training data for rank {rank}")
        # determine the indices of the columns to be normalized
        normalize_indices = self.get_normalize_indices(X_train_local)

        # feature normalization
        if X_train_local.size:
            local_feature_sum = np.sum(X_train_local[:, normalize_indices], axis=0)
        else:
            local_feature_sum = np.zeros(len(normalize_indices))

        global_feature_sum = comm.allreduce(local_feature_sum, op=MPI.SUM)

        local_feature_count = X_train_local.shape[0]
        global_feature_count = comm.allreduce(local_feature_count, op=MPI.SUM)

        self.x_train_mean = global_feature_sum / global_feature_count if global_feature_count > 0 else np.zeros_like(global_feature_sum)

        if X_train_local.size:
            local_sqdiff = np.sum((X_train_local[:, normalize_indices] - self.x_train_mean) ** 2, axis=0)
        else:
            local_sqdiff = np.zeros(len(normalize_indices))

        global_sqdiff = comm.allreduce(local_sqdiff, op=MPI.SUM)
        self.x_train_std = np.sqrt(global_sqdiff / global_feature_count) if global_feature_count > 0 else np.ones_like(global_sqdiff)

        # normalize in place, only for normalize_indices
        if X_train_local.size > 0:
            x_train_std_nozero = np.where(self.x_train_std == 0, 1.0, self.x_train_std) # covering edge case of stddev being 0
            X_train_local[:, normalize_indices] = (X_train_local[:, normalize_indices] - self.x_train_mean) / x_train_std_nozero

        # label normalization
        local_label_sum = float(np.sum(y_train_local)) if y_train_local.size else 0.0
        global_label_sum = comm.allreduce(local_label_sum, op=MPI.SUM)

        local_label_count = y_train_local.shape[0]
        global_label_count = comm.allreduce(local_label_count, op=MPI.SUM)

        self.y_train_mean = global_label_sum / global_label_count if global_label_count > 0 else 0.0

        local_sqdiff = float(np.sum((y_train_local - self.y_train_mean) ** 2)) if y_train_local.size else 0.0
        global_sqdiff = comm.allreduce(local_sqdiff, op=MPI.SUM)
        self.y_train_std = np.sqrt(global_sqdiff / global_label_count) if global_label_count > 0 else 1.0

        if y_train_local.size > 0:
            std_nozero = self.y_train_std if self.y_train_std > 0 else 1.0
            y_train_local[:] = (y_train_local - self.y_train_mean) / std_nozero
        logger.info(f"Finished training normalization for rank {rank}")
        return X_train_local, y_train_local


    def normalize_test_data(self, x_test, y_test):
        """Normalize the test data using the global means and standard deviations from the training data."""
        logger.info(f"Starting normalization of test data for rank {rank}")
        normalize_indices = self.get_normalize_indices(x_test)

        x_test_normalized = x_test.copy()
        if x_test.size and normalize_indices:
            x_train_std_nozero = np.where(self.x_train_std == 0, 1.0, self.x_train_std)
            x_test_normalized[:, normalize_indices] = (x_test[:, normalize_indices] - self.x_train_mean) / x_train_std_nozero

        y_test_normalized = y_test.copy()
        if y_test.size:
            y_train_std_nozero = self.y_train_std if self.y_train_std > 0 else 1.0
            y_test_normalized[:] = (y_test - self.y_train_mean) / y_train_std_nozero
        logger.info(f"Finished normalization of test data for rank {rank}")
        return x_test_normalized, y_test_normalized

    def get_normalize_indices(self, feature_matrix):
        # get number of features
        n_feature = feature_matrix.shape[1] if feature_matrix.size else 0

        # determine the indices of the columns to be normalized
        skip_indices = []
        for col in self.skip_normalization_columns:
            if col in self.feature_columns:
                skip_indices.append(self.feature_columns.index(col))
        
        return [i for i in range(n_feature) if i not in skip_indices]