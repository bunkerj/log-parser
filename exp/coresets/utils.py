import numpy as np
from copy import deepcopy
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from exp.mixture_models.utils import get_num_true_clusters
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline
from src.coresets.greedy_iterative_geodesic_ascent import \
    GreedyIterativeGeodesicAscent


def run_coreset_exp(proj_dim, subset_size, n_samples, data_config):
    data_manager = DataManager(data_config)
    tokenized_logs = data_manager.get_tokenized_logs()

    true_assignments = data_manager.get_true_assignments()
    num_clusters = get_num_true_clusters(true_assignments)
    evaluator = Evaluator(true_assignments)

    score_samples_online = []
    score_samples_coreset = []
    coreset_sizes = []
    for _ in range(n_samples):
        mmo_online = MultinomialMixtureOnline(tokenized_logs,
                                              num_clusters,
                                              is_classification=False,
                                              alpha=1.05,
                                              beta=1.05)
        mmo_coreset = deepcopy(mmo_online)
        mmo_online.perform_online_batch_em(tokenized_logs)

        cluster_pos = np.array(mmo_online.t_c)
        vocab_pos = np.array(mmo_online.t_v)

        geo_ascent = GreedyIterativeGeodesicAscent(tokenized_logs,
                                                   cluster_pos=cluster_pos,
                                                   vocab_pos=vocab_pos,
                                                   num_clusters=num_clusters)
        reduced_weights, reduced_set = geo_ascent.get_coreset(subset_size,
                                                              proj_dim)

        mmo_online_clusters = mmo_online.get_clusters(tokenized_logs)
        score_online = evaluator.get_nmi(mmo_online_clusters)

        mmo_coreset.perform_offline_em(reduced_set, weights=reduced_weights)
        mmo_coreset_clusters = mmo_coreset.get_clusters(tokenized_logs)
        score_coreset = evaluator.get_nmi(mmo_coreset_clusters)

        score_samples_online.append(score_online)
        score_samples_coreset.append(score_coreset)
        coreset_sizes.append(len(reduced_weights))

    return {
        'score_samples_online': score_samples_online,
        'score_samples_coreset': score_samples_coreset,
        'coreset_sizes': coreset_sizes,
    }
