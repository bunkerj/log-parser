import numpy as np


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
    distances = np.sqrt(((matrix - mean) ** 2).sum(axis=1))
    return distances.mean()


def get_intra_cluster_spread(count_per_cluster_split):
    """
    Returns a weighted average of the average distances to the mean vector for
    each cluster.
    """
    weighted_avg_distance_to_mean = 0
    for C_cluster in count_per_cluster_split.values():
        avg_dist_to_mean = get_average_row_distances_from_mean(C_cluster)
        weighted_avg_distance_to_mean += avg_dist_to_mean / C_cluster.shape[0]
    return weighted_avg_distance_to_mean


def get_inter_cluster_spread(count_per_cluster_split):
    """
    Returns the average distance between the means of all clusters with respect
    to the mean of means.
    """
    C_means = None
    for C_cluster in count_per_cluster_split.values():
        cluster_mean = C_cluster.mean(axis=0, keepdims=True)
        C_means = concatenate_row(C_means, cluster_mean)
    return get_average_row_distances_from_mean(C_means)
