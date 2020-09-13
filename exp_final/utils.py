import numpy as np
from src.parsers.multinomial_mixture_vb import MultinomialMixtureVB
from src.coresets.greedy_iterative_geodesic_ascent import \
    GreedyIterativeGeodesicAscent


def get_log_sample(data_manager, n_samples):
    p_indices = np.random.permutation(n_samples)

    logs = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()

    sample_logs = [logs[idx] for idx in p_indices]
    samples_true_assign = [true_assignments[idx] for idx in p_indices]

    return sample_logs, samples_true_assign


def get_posterior_approx(logs, n_clusters):
    mm = MultinomialMixtureVB()
    mm.fit(logs, n_clusters, max_iter=1)

    cluster_pos = mm.pi_v
    vocab_pos = mm.theta_v

    return cluster_pos, vocab_pos


def get_coreset(logs, n_clusters, subset_size, proj_dim):
    # cluster_pos, vocab_pos = get_posterior_approx(logs, n_clusters)
    geo_ascent = GreedyIterativeGeodesicAscent(logs, num_clusters=n_clusters)
                                               # cluster_pos=cluster_pos,
                                               # vocab_pos=vocab_pos)
    return geo_ascent.get_coreset(subset_size, proj_dim)


def get_log_strings(logs, c_base):
    c_base_strings = {}
    for k in c_base:
        c_base_strings[k] = []
        for idx in c_base[k]:
            log_str = ' '.join(logs[idx])
            c_base_strings[k].append(log_str)
    return c_base_strings
