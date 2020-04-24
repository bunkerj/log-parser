import numpy as np
from numpy.linalg import norm
from scipy.stats import entropy


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


def get_sa_problem(X_df):
    X_info_df = X_df.describe()

    parameter_ranges_dict = {}
    for col in X_info_df.columns:
        parameter_ranges_dict[col] = (
            X_info_df[col]['min'], X_info_df[col]['max'])

    return {
        'num_vars': len(parameter_ranges_dict),
        'names': list(parameter_ranges_dict.keys()),
        'bounds': [parameter_ranges_dict[k] for k in parameter_ranges_dict],
    }


def get_avg_entropy(matrix, axis):
    entropy_values = np.apply_along_axis(entropy, axis, matrix)
    return entropy_values.mean()


def get_sum_entropy(matrix):
    sum_vector = matrix.sum(axis=0)
    return get_avg_entropy(sum_vector, 0)


def get_flat_entropy(matrix):
    probabilities_vector = matrix.flatten()
    return get_avg_entropy(probabilities_vector, 0)
