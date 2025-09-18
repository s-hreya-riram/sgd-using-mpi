import numpy as np
from processing.mpi_utils import *

class Normalizer:
    def __init__(self, feature_columns, skip_normalization_columns):
        self.feature_columns = feature_columns
        self.skip_normalization_columns = skip_normalization_columns

        # attributes after normalizing train data that will be used for normalizing test data
        self.feature_mean = None
        self.feature_std = None
        self.label_mean = None
        self.label_std = None

    def normalize(self, training_feature_local, training_label_local):
        """Following the same process as professor did in Kernel Ridge Regression,
        calculating local sum/sq diff and then using allreduce to get global sum/sqdiff."""
        
        # determine the indices of the columns to be normalized
        normalize_indices = self.get_normalize_indices(training_feature_local)

        # feature normalization
        if training_feature_local.size:
            local_feature_sum = np.sum(training_feature_local[:, normalize_indices], axis=0)
        else:
            local_feature_sum = np.zeros(len(normalize_indices))

        global_feature_sum = comm.allreduce(local_feature_sum, op=MPI.SUM)

        local_feature_count = training_feature_local.shape[0]
        global_feature_count = comm.allreduce(local_feature_count, op=MPI.SUM)

        self.feature_mean = global_feature_sum / global_feature_count if global_feature_count > 0 else np.zeros_like(global_feature_sum)

        if training_feature_local.size:
            local_sqdiff = np.sum((training_feature_local[:, normalize_indices] - self.feature_mean) ** 2, axis=0)
        else:
            local_sqdiff = np.zeros(len(normalize_indices))

        global_sqdiff = comm.allreduce(local_sqdiff, op=MPI.SUM)
        self.feature_std = np.sqrt(global_sqdiff / global_feature_count) if global_feature_count > 0 else np.ones_like(global_sqdiff)

        # normalize in place, only for normalize_indices
        if training_feature_local.size > 0:
            feature_std_nozero = np.where(self.feature_std == 0, 1.0, self.feature_std) # covering edge case of stddev being 0
            training_feature_local[:, normalize_indices] = (training_feature_local[:, normalize_indices] - self.feature_mean) / feature_std_nozero

        # label normalization
        local_label_sum = float(np.sum(training_label_local)) if training_label_local.size else 0.0
        global_label_sum = comm.allreduce(local_label_sum, op=MPI.SUM)

        local_label_count = training_label_local.shape[0]
        global_label_count = comm.allreduce(local_label_count, op=MPI.SUM)

        self.label_mean = global_label_sum / global_label_count if global_label_count > 0 else 0.0

        local_sqdiff = float(np.sum((training_label_local - self.label_mean) ** 2)) if training_label_local.size else 0.0
        global_sqdiff = comm.allreduce(local_sqdiff, op=MPI.SUM)
        self.label_std = np.sqrt(global_sqdiff / global_label_count) if global_label_count > 0 else 1.0

        if training_label_local.size > 0:
            std_nozero = self.label_std if self.label_std > 0 else 1.0
            training_label_local = (training_label_local - self.label_mean) / std_nozero

        return training_feature_local, training_label_local


    def normalize_test_data(self, test_feature, test_label):
        """Normalize the test data using the means and standard deviations from the training data."""
        normalize_indices = self.get_normalize_indices(test_feature)

        test_feature_normalized = test_feature.copy()
        if test_feature.size and normalize_indices:
            feature_std_nozero = np.where(self.feature_std == 0, 1.0, self.feature_std)
            test_feature_normalized[:, normalize_indices] = (test_feature[:, normalize_indices] - self.feature_mean) / feature_std_nozero

        test_label_normalized = test_label.copy()
        if test_label.size:
            label_std_nozero = self.label_std if self.label_std > 0 else 1.0
            test_label_normalized = (test_label - self.label_mean) / label_std_nozero

        return test_feature_normalized, test_label_normalized

    def get_normalize_indices(self, feature_matrix):
        # calculating number of features
        n_feature = feature_matrix.shape[1] if feature_matrix.size else 0

        # determine the indices of the columns to be normalized
        skip_indices = []
        for col in self.skip_normalization_columns:
            if col in self.feature_columns:
                skip_indices.append(self.feature_columns.index(col))
        
        return [i for i in range(n_feature) if i not in skip_indices]