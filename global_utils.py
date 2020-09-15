import os
import pickle
import numpy as np
from random import shuffle
from scipy.special import gammaln, xlogy
from global_constants import RESULTS_DIR


def dump_results(name, results, results_dir=None):
    path = create_file_path(name, results_dir)
    pickle.dump(results, open(path, 'wb'))


def load_results(name, results_dir=RESULTS_DIR):
    path = os.path.join(results_dir, name)
    return pickle.load(open(path, 'rb'))


def create_file_path(name, results_dir):
    results_dir = RESULTS_DIR if results_dir is None else results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    path = os.path.join(results_dir, name)
    return path


def shuffle_same_order(*arrays):
    c = list(zip(*arrays))
    shuffle(c)
    return list(zip(*c))


def log_multi_coeff(x_flat):
    return gammaln(x_flat.sum() + 1) - np.sum(gammaln(x_flat + 1))


def unnorm_log_multi(x_flat, params):
    params_flat = np.maximum(params, 0).flatten()
    return (x_flat * np.log(params_flat)).sum()


def log_multi(x, params):
    x_flat = x.flatten()
    params_flat = np.maximum(params, 0).flatten()
    return log_multi_coeff(x_flat) + xlogy(x_flat, params_flat).sum()


def multi(x, params):
    return np.exp(log_multi(x, params))


def get_multi_values(x, theta):
    x_n_flat = x.flatten()
    num_clusters = theta.shape[0]
    multi_values = np.zeros(num_clusters)
    log_coeff = log_multi_coeff(x_n_flat)
    for g in range(num_clusters):
        log_multi_g = log_coeff + unnorm_log_multi(x_n_flat, theta[g, :])
        multi_values[g] = np.exp(log_multi_g)
    return multi_values


def log_multi_beta(alpha):
    sum_axis = alpha.ndim - 1
    return gammaln(alpha).sum(axis=sum_axis) - gammaln(alpha.sum(axis=sum_axis))


def get_top_k_args(arr, k):
    return arr.flatten().argsort()[-k:][::-1]


def get_avg(samples):
    n_samples = len(samples)
    sample_len = len(samples[0])
    averages = []
    for label_idx in range(sample_len):
        avg_lab_impurity = 0
        for sample_idx in range(n_samples):
            avg_lab_impurity += samples[sample_idx][label_idx]
        averages.append(avg_lab_impurity / n_samples)
    return averages


def get_labeled_indices(log_labels):
    labeled_indices = []
    for k in log_labels:
        labeled_indices.extend(log_labels[k])
    return labeled_indices
