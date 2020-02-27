import numpy as np
from random import sample


def get_log_labels(true_assignments, num_of_labels):
    log_labels = {}
    labeled_indices = sample(range(len(true_assignments)), k=num_of_labels)
    for log_idx in labeled_indices:
        cluster = true_assignments[log_idx][-1]
        if cluster not in log_labels:
            log_labels[cluster] = []
        log_labels[cluster].append(log_idx)
    return log_labels


def get_num_true_clusters(true_assignments):
    return len(set(log_data[-1] for log_data in true_assignments))


def normalize_vector(vector):
    n = sum(vector)
    if n == 0:
        return vector
    else:
        return vector / n


def normalize_matrix(matrix, axis):
    return np.apply_along_axis(normalize_vector, axis, matrix)


def get_gini_impurity(probabilities):
    return sum(probabilities * (1 - probabilities))


def get_avg_gini_impurity(probabilities_matrix, axis):
    entropy_values = np.apply_along_axis(get_gini_impurity,
                                         axis,
                                         probabilities_matrix)
    return sum(entropy_values) / len(entropy_values)


def get_impurity_difference(labeled_impurity, unlabeled_impurity):
    abs_diff = abs(labeled_impurity - unlabeled_impurity)
    return 100 * abs_diff / labeled_impurity
