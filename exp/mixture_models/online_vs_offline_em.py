"""
Log-likelihood comparison between online and offline EM.
"""
from copy import deepcopy
from global_utils import dump_results, get_num_true_clusters
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline


def run_online_vs_offline_em(data_configs, n_init):
    results = {}
    for data_config in data_configs:
        name = data_config['name']
        results[name] = {'offline': None, 'online': None}

        print('{}...'.format(name))

        data_manager = DataManager(data_config)
        logs = data_manager.get_tokenized_logs()
        true_assignments = data_manager.get_true_assignments()
        n_true_clusters = get_num_true_clusters(true_assignments)

        online_em_parser = MultinomialMixtureOnline(logs,
                                                    n_true_clusters,
                                                    False,
                                                    epsilon=0.01,
                                                    alpha=1.05,
                                                    beta=1.05)
        online_em_parser.find_best_initialization(logs, n_init=n_init)
        offline_em_parser = deepcopy(online_em_parser)

        offline_em_parser.perform_offline_em(logs, track_history=True)
        offline_ll_history = offline_em_parser.get_log_likelihood_history()
        results[name]['offline'] = offline_ll_history

        online_em_parser.perform_online_batch_em(logs,
                                                 len(offline_ll_history) - 1,
                                                 track_history=True)
        online_ll_history = online_em_parser.get_log_likelihood_history()
        results[name]['online'] = online_ll_history

    print('Done!')

    return results


if __name__ == '__main__':
    n_init = 200
    data_configs = [
        DataConfigs.Android,
        DataConfigs.Apache,
        DataConfigs.BGL,
        DataConfigs.Hadoop,
        DataConfigs.HDFS,
        DataConfigs.HealthApp,
        DataConfigs.HPC,
        DataConfigs.Linux,
    ]

    results = run_online_vs_offline_em(data_configs, n_init)
    dump_results('online_vs_offline_em.p', results)
