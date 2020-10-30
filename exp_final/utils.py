import numpy as np
from src.parsers.multinomial_mixture_vb import MultinomialMixtureVB
from src.coresets.greedy_iterative_geodesic_ascent import \
    GreedyIterativeGeodesicAscent


def get_log_sample(data_manager, n_samples):
    logs = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()
    p_indices = np.random.permutation(len(logs))

    p_logs = [logs[idx] for idx in p_indices]
    p_true_assign = [true_assignments[idx] for idx in p_indices]

    return p_logs[:n_samples], p_true_assign[:n_samples]


def permute_logs(logs, true_assignments):
    n = len(logs)
    logs_permuted = []
    true_assignments_permuted = []
    for idx in np.random.permutation(n):
        log = logs[idx]
        true_assignment = true_assignments[idx]
        logs_permuted.append(log)
        true_assignments_permuted.append(true_assignment)
    return logs_permuted, true_assignments_permuted


def get_posterior_approx(logs, n_clusters):
    mm = MultinomialMixtureVB()
    mm.fit(logs, n_clusters, max_iter=1)

    cluster_pos = mm.pi_v
    vocab_pos = mm.theta_v

    return cluster_pos, vocab_pos


def get_coreset(logs, n_clusters, subset_size, proj_dim):
    cluster_pos, vocab_pos = get_posterior_approx(logs, n_clusters)
    geo_ascent = GreedyIterativeGeodesicAscent(logs,
                                               cluster_pos=cluster_pos + 1,
                                               vocab_pos=vocab_pos + 1)
    return geo_ascent.get_coreset(subset_size, proj_dim)


def get_log_str(logs, idx):
    return ' '.join(logs[idx])


def get_log_strings(logs, c_base):
    c_base_strings = {}
    for k in c_base:
        c_base_strings[k] = []
        for idx in c_base[k]:
            log_str = get_log_str(logs, idx)
            c_base_strings[k].append(log_str)
    return c_base_strings


def get_constraint_strings(logs, constraints):
    constraint_strings = {}
    for k in constraints:
        constraint_strings[k] = []
        for link in constraints[k]:
            idx1, idx2 = link
            log_str1 = get_log_str(logs, idx1)
            log_str2 = get_log_str(logs, idx2)
            constraint_strings[k].append((log_str1, log_str2))
    return constraint_strings
