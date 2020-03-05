import numpy as np


def concatenate_row(matrix, array):
    row = array.reshape(1, -1)
    if matrix is None:
        return row
    else:
        return np.concatenate([matrix, row])


def split_counts_per_cluster(C, true_assignments):
    count_cluster_split = {}
    for idx in range(C.shape[0]):
        C_cluster = None
        cluster = true_assignments[idx][-2]
        if cluster in count_cluster_split:
            C_cluster = count_cluster_split[cluster]
        count_cluster_split[cluster] = concatenate_row(C_cluster, C[idx, :])
    return count_cluster_split


def get_mean_squared_error(matrix):
    """
    1) Compute the mean vector across all row.
    2) For each row, compute the distance between that row and the
       mean vector.
    3) Return the average of all of these distances.
    """
    mean = matrix.mean(axis=0, keepdims=True)
    distances = np.sqrt(((matrix - mean) ** 2).sum(axis=1))
    return distances.mean()


def get_avg_intra_score(count_per_cluster_split):
    avg_intra_cluster_score = 0
    for C_cluster in count_per_cluster_split.values():
        intra_cluster_score = get_mean_squared_error(C_cluster)
        avg_intra_cluster_score += intra_cluster_score / C_cluster.shape[0]
    return avg_intra_cluster_score


def get_avg_inter_score(count_per_cluster_split):
    C_means = None
    for C_cluster in count_per_cluster_split.values():
        cluster_mean = C_cluster.mean(axis=0, keepdims=True)
        C_means = concatenate_row(C_means, cluster_mean)
    return get_mean_squared_error(C_means)
