import numpy as np
from processing.mpi_utils import *

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
