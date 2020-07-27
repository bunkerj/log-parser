import numpy as np
import multiprocessing as mp
from copy import deepcopy
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from exp.mixture_models.utils import get_num_true_clusters
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline
from src.coresets.greedy_iterative_geodesic_ascent import \
    GreedyIterativeGeodesicAscent


def run_coreset_exp_mp(proj_dim, subset_size, n_samples, data_config):
    data_manager = DataManager(data_config)
    tokenized_logs = data_manager.get_tokenized_logs()

    true_assignments = data_manager.get_true_assignments()
    num_clusters = get_num_true_clusters(true_assignments)
    evaluator = Evaluator(true_assignments)

    with mp.Pool(mp.cpu_count()) as pool:
        arguments = [(evaluator, num_clusters, proj_dim,
                      subset_size, tokenized_logs) for _ in range(n_samples)]
        mp_results = pool.starmap(run_single_coreset_exp, arguments)
        score_samples_coreset, score_samples_online = list(zip(*mp_results))

    return {
        'score_samples_online': score_samples_online,
        'score_samples_coreset': score_samples_coreset,
    }


def run_single_coreset_exp(evaluator, num_clusters, proj_dim, subset_size,
                           tokenized_logs):
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

    return score_coreset, score_online