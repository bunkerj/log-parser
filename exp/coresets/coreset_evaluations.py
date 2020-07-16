import numpy as np
from copy import deepcopy
from src.coresets.greedy_iterative_geodesic_ascent import \
    GreedyIterativeGeodesicAscent
from exp.mixture_models.utils import get_num_true_clusters
from global_utils import dump_results
from src.coresets.random_vector_projector import RandomVectorProjector
from src.helpers.data_manager import DataManager
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline
from src.utils import get_vocabulary_indices, get_token_counts


def run_coreset_evaluations(proj_vector_dim, data_config):
    data_manager = DataManager(data_config)
    tokenized_logs = data_manager.get_tokenized_logs()

    v_indices = get_vocabulary_indices(tokenized_logs)
    num_vocab = len(v_indices)

    true_assignments = data_manager.get_true_assignments()
    evaluator = Evaluator(true_assignments)
    num_true_clusters = get_num_true_clusters(true_assignments)
    log_vectors = [get_token_counts(log, v_indices) for log in tokenized_logs]

    cluster_posterior = np.ones(num_true_clusters)
    vocab_posterior = np.ones((num_true_clusters, num_vocab))

    vector_projector = RandomVectorProjector(num_true_clusters, num_vocab,
                                             log_vectors, cluster_posterior,
                                             vocab_posterior, proj_vector_dim)

    projections = vector_projector.get_fw_projections()
    geo_ascent = GreedyIterativeGeodesicAscent(projections, 50)
    reduced_weights, reduced_set = geo_ascent.get_coreset(tokenized_logs)

    mmo = MultinomialMixtureOnline(tokenized_logs,
                                   num_true_clusters,
                                   alpha=1.05,
                                   beta=1.05)
    mmo_coreset = deepcopy(mmo)

    mmo.perform_offline_em(tokenized_logs)
    mmo_coreset.perform_offline_em(reduced_set, weights=reduced_weights)

    mmo_clusters = mmo.get_clusters(tokenized_logs)
    mmo_coreset_clusters = mmo_coreset.get_clusters(tokenized_logs)

    return {
        'mmo_score': evaluator.get_impurity(mmo_clusters, []),
        'mmo_coreset_score': evaluator.get_impurity(mmo_coreset_clusters, []),
        'reduced_weights': reduced_weights,
        'reduced_set': reduced_set,
    }


if __name__ == '__main__':
    data_config = DataConfigs.HDFS
    proj_vector_dim = 50
    results = run_coreset_evaluations(proj_vector_dim, data_config)
    print(results)
    dump_results('coreset_evaluations.p', results)
