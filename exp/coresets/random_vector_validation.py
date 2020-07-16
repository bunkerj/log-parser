import numpy as np
import functools
from math import sqrt
from src.coresets.greedy_iterative_geodesic_ascent import \
    GreedyIterativeGeodesicAscent
from exp.mixture_models.utils import get_num_true_clusters
from src.coresets.random_vector_projector import RandomVectorProjector
from src.helpers.data_manager import DataManager
from src.data_config import DataConfigs
from src.utils import get_vocabulary_indices, get_token_counts


def run_random_vector_validation(proj_vector_dim, data_config):
    data_manager = DataManager(data_config)
    tokenized_logs = data_manager.get_tokenized_logs()

    v_indices = get_vocabulary_indices(tokenized_logs)
    num_vocab = len(v_indices)

    true_assignments = data_manager.get_true_assignments()
    num_true_clusters = get_num_true_clusters(true_assignments)
    log_vectors = [get_token_counts(log, v_indices) for log in tokenized_logs]

    cluster_posterior = np.ones(num_true_clusters)
    vocab_posterior = np.ones((num_true_clusters, num_vocab))

    vector_projector = RandomVectorProjector(num_true_clusters, num_vocab,
                                             log_vectors, cluster_posterior,
                                             vocab_posterior, proj_vector_dim)

    return vector_projector.get_fw_projections()


if __name__ == '__main__':
    data_config = DataConfigs.Apache
    proj_vector_dim = 100
    projections = run_random_vector_validation(proj_vector_dim, data_config)
    matrix = np.stack([v_n.flatten() for v_n in projections], axis=1)

    L = functools.reduce(lambda a, b: a + b, projections)
    for m in range(1, 21):
        geo_ascent = GreedyIterativeGeodesicAscent(projections, m)
        w = geo_ascent.get_weights().reshape((-1, 1))
        L_approx = matrix @ w
        diff_v = L - L_approx
        diff = sqrt(diff_v.T @ diff_v)
        print('{:<5}: {:<10}'.format(m, diff))
