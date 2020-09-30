"""
Compare unlabeled (baseline) impurities against labeled impurities for the
online mixture model.
"""
from copy import deepcopy
from global_utils import dump_results, sample_log_labels, get_num_true_clusters
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline


def run_online_benchmark(n_labels):
    data_configs = [
        DataConfigs.Android,
        DataConfigs.Apache,
        DataConfigs.BGL,
        DataConfigs.Hadoop,
        DataConfigs.HDFS,
        DataConfigs.HealthApp,
        DataConfigs.HPC,
        DataConfigs.Linux,
        DataConfigs.Mac,
        DataConfigs.OpenSSH,
        DataConfigs.OpenStack,
        DataConfigs.Proxifier,
        DataConfigs.Spark,
        DataConfigs.Thunderbird,
        DataConfigs.Windows,
        DataConfigs.Zookeeper,
    ]

    results = {}
    for data_config in data_configs:
        name = data_config['name']
        print('{}...'.format(name))

        data_manager = DataManager(data_config)
        logs = data_manager.get_tokenized_logs()
        true_assignments = data_manager.get_true_assignments()
        ev = Evaluator(true_assignments)

        n_true_clusters = get_num_true_clusters(true_assignments)
        log_labels = sample_log_labels(true_assignments, n_labels)

        parser_lab = MultinomialMixtureOnline(logs,
                                              n_true_clusters,
                                              is_classification=False,
                                              alpha=1.05,
                                              beta=1.05)
        parser_lab.find_best_initialization(logs, 100, 200)
        parser_unlab = deepcopy(parser_lab)

        parser_lab.label_logs(log_labels, logs)
        parser_lab.perform_online_batch_em(logs)
        parser_unlab.perform_online_batch_em(logs)

        c_lab = parser_lab.get_clusters(logs)
        c_unlab = parser_unlab.get_clusters(logs)

        impurities_lab = ev.get_impurity(c_lab, parser_lab.labeled_indices)
        impurities_unlab = ev.get_impurity(c_unlab, parser_lab.labeled_indices)

        results[name] = {'lab': impurities_lab, 'unlab': impurities_unlab}

    return results


if __name__ == '__main__':
    n_labels = 200

    results = run_online_benchmark(n_labels)
    dump_results('online_benchmark_test.p', results)
