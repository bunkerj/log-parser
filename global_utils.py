import os
import pickle
import numpy as np
from collections import defaultdict
from random import shuffle, choices
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


def get_log_labels(true_assignments):
    log_labels = defaultdict(list)
    for assignment in true_assignments:
        log_idx = int(assignment[0])
        event = assignment[12]
        log_labels[event].append(log_idx)
    return log_labels


def sample_log_labels(true_assignments, num_of_labels):
    log_labels = {}
    labeled_indices = choices(range(len(true_assignments)), k=num_of_labels)
    for log_idx in labeled_indices:
        cluster = true_assignments[log_idx][-1]
        if cluster not in log_labels:
            log_labels[cluster] = []
        log_labels[cluster].append(log_idx)
    return log_labels


def get_num_true_clusters(true_assignments):
    return len(set(log_data[-1] for log_data in true_assignments))


def get_parsed_events(true_assignments, parsed_clusters):
    parsed_reference = [0] * len(true_assignments)
    for event_idx, event in enumerate(parsed_clusters):
        for log_idx in parsed_clusters[event]:
            parsed_reference[log_idx] = event_idx
    return parsed_reference


def get_constraints_info(logs, true_assignments, parsed_clusters, W):
    link_constraints = {'must_link': [], 'cannot_link': []}
    parsed_events = get_parsed_events(true_assignments, parsed_clusters)
    for i in W:
        log1 = ' '.join(logs[i])
        g1_true = true_assignments[i][-2]
        g1_parsed = parsed_events[i]
        for j in W[i]:
            log2 = ' '.join(logs[j])
            g2_true = true_assignments[j][-2]
            g2_parsed = parsed_events[j]
            w = W[i][j]
            v = ((i, log1, g1_true, g1_parsed),
                 (j, log2, g2_true, g2_parsed))
            if w > 0:
                link_constraints['must_link'].append(v)
            else:
                link_constraints['cannot_link'].append(v)
    return link_constraints


def get_link_events(links, reference):
    link_events = []
    for link in links:
        idx1, idx2 = link
        event1 = reference[idx1]
        event2 = reference[idx2]
        link_events.append((event1, event2))
    return link_events
