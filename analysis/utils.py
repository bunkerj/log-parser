import numpy as np
from numpy.linalg import norm


def concatenate_row(matrix, array):
    row = array.reshape(1, -1)
    if matrix is None:
        return row
    else:
        return np.concatenate([matrix, row])


def split_counts_per_cluster(C, true_assignments):
    """
    Returns a dictionary where each key represents a cluster and the value is a
    matrix containing, as a row, the vector representation of each log.
    """
    count_cluster_split = {}
    for idx in range(C.shape[0]):
        cluster = true_assignments[idx][-2]
        C_cluster = count_cluster_split.get(cluster)
        count_cluster_split[cluster] = concatenate_row(C_cluster, C[idx, :])
    return count_cluster_split


def get_average_row_distances_from_mean(matrix):
    """
    1) Compute the mean vector across all row.
    2) For each row, compute the distance between that row and the
       mean vector.
    3) Return the average of all of these distances.
    """
    mean = matrix.mean(axis=0, keepdims=True)
    distances = norm(matrix - mean, axis=1)
    return distances.mean()


def get_intra_cluster_spread(count_per_cluster_split, C):
    """
    Returns a weighted average of the average distances to the mean vector for
    each cluster.
    """
    total_count = C.shape[0]
    weighted_avg_distance_to_mean = 0
    for C_cluster in count_per_cluster_split.values():
        avg_dist_to_mean = get_average_row_distances_from_mean(C_cluster)
        weighted_avg_distance_to_mean += avg_dist_to_mean * C_cluster.shape[0]
    return weighted_avg_distance_to_mean / total_count


def get_inter_cluster_spread(count_per_cluster_split, C):
    """
    Returns a weighted average distance between the means of every clusters with
    respect to the global mean.
    """
    total_count = C.shape[0]
    weighted_avg_distance_to_mean = 0
    global_mean = C.mean(axis=0, keepdims=True)
    for C_cluster in count_per_cluster_split.values():
        cluster_mean = C_cluster.mean(axis=0, keepdims=True)
        distance = float(norm(global_mean - cluster_mean, axis=1))
        weighted_avg_distance_to_mean += distance * C_cluster.shape[0]
    return weighted_avg_distance_to_mean / total_count


def get_avg_from_samples(samples):
    return np.array(samples).mean(axis=0)


def get_var_from_samples(samples):
    return np.array(samples).var(ddof=1, axis=0)


def get_labels_from_true_assignments(true_assignments):
    return [ta[-2] for ta in true_assignments]
