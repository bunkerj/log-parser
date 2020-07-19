"""
Compare performance of offline EM with and without the coreset.
"""
import numpy as np
from copy import deepcopy
from src.coresets.greedy_iterative_geodesic_ascent import \
    GreedyIterativeGeodesicAscent
from exp.mixture_models.utils import get_num_true_clusters
from global_utils import dump_results
from src.helpers.data_manager import DataManager
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline


def run_coreset_evaluations(proj_dim, subset_size, n_samples, data_config):
    data_manager = DataManager(data_config)
    tokenized_logs = data_manager.get_tokenized_logs()

    true_assignments = data_manager.get_true_assignments()
    evaluator = Evaluator(true_assignments)
    num_true_clusters = get_num_true_clusters(true_assignments)

    mmo_pos_init = MultinomialMixtureOnline(tokenized_logs, num_true_clusters,
                                            alpha=1.05,
                                            beta=1.05)
    mmo_pos_init.perform_online_batch_em(tokenized_logs)

    cluster_pos = np.array(mmo_pos_init.t_c)
    vocab_pos = np.array(mmo_pos_init.t_v)

    geo_ascent = GreedyIterativeGeodesicAscent(tokenized_logs,
                                               # cluster_pos=cluster_pos,
                                               # vocab_pos=vocab_pos,
                                               num_clusters=num_true_clusters)
    reduced_weights, reduced_set = geo_ascent.get_coreset(subset_size, proj_dim)

    score_samples_offline = []
    score_samples_coreset = []
    score_samples_online = []
    for _ in range(n_samples):
        mmo = MultinomialMixtureOnline(tokenized_logs, num_true_clusters,
                                       alpha=1.05,
                                       beta=1.05)
        mmo_coreset = deepcopy(mmo)
        mmo_online = deepcopy(mmo)

        mmo.perform_offline_em(tokenized_logs)
        mmo_clusters = mmo.get_clusters(tokenized_logs)
        score_offline = evaluator.get_nmi(mmo_clusters)

        mmo_coreset.perform_offline_em(reduced_set, weights=reduced_weights)
        mmo_coreset_clusters = mmo_coreset.get_clusters(tokenized_logs)
        score_coreset = evaluator.get_nmi(mmo_coreset_clusters)

        mmo_online.perform_online_batch_em(tokenized_logs)
        mmo_online_clusters = mmo_online.get_clusters(tokenized_logs)
        score_online = evaluator.get_nmi(mmo_online_clusters)

        score_samples_offline.append(score_offline)
        score_samples_coreset.append(score_coreset)
        score_samples_online.append(score_online)

    return {
        'score_samples_offline': score_samples_offline,
        'score_samples_coreset': score_samples_coreset,
        'score_samples_online': score_samples_online,
        'coreset_size': len(reduced_weights),
    }


if __name__ == '__main__':
    proj_dim = 500
    subset_size = 50
    n_samples = 25

    data_configs = [
        DataConfigs.Android,
        DataConfigs.Apache,
        DataConfigs.BGL,
        DataConfigs.Hadoop,
        DataConfigs.HDFS,
        DataConfigs.HealthApp,
        DataConfigs.HPC,
        DataConfigs.Linux,
        # DataConfigs.Mac,
        # DataConfigs.OpenSSH,
        # DataConfigs.OpenStack,
        # DataConfigs.Proxifier,
        # DataConfigs.Spark,
        # DataConfigs.Thunderbird,
        # DataConfigs.Windows,
        # DataConfigs.Zookeeper,
    ]

    for data_config in data_configs:
        name = data_config['name'].lower()
        print(name)
        results = run_coreset_evaluations(proj_dim, subset_size, n_samples,
                                          data_config)
        filename = 'coreset_evaluations_{}_nmi_pos.p'.format(name)
        dump_results(filename, results)
