"""
Compare performance of offline EM with and without the coreset.
"""
from copy import deepcopy
from src.coresets.greedy_iterative_geodesic_ascent import \
    GreedyIterativeGeodesicAscent
from exp.mixture_models.utils import get_num_true_clusters
from global_utils import dump_results
from src.helpers.data_manager import DataManager
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline


def run_coreset_evaluations(proj_dim, data_config):
    data_manager = DataManager(data_config)
    tokenized_logs = data_manager.get_tokenized_logs()

    true_assignments = data_manager.get_true_assignments()
    evaluator = Evaluator(true_assignments)
    num_true_clusters = get_num_true_clusters(true_assignments)

    geo_ascent = GreedyIterativeGeodesicAscent(tokenized_logs,
                                               num_clusters=num_true_clusters)
    reduced_weights, reduced_set = geo_ascent.get_coreset(50, proj_dim)

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
        'coreset_size': len(reduced_weights),
    }


if __name__ == '__main__':
    data_config = DataConfigs.HealthApp
    proj_dim = 50
    results = run_coreset_evaluations(proj_dim, data_config)
    for k in results:
        print('{:<10} {:<10}'.format(k, results[k]))
    dump_results('coreset_evaluations.p', results)
