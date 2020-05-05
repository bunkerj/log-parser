"""
Compare online mixture model with random initialization against online mixture
model with Drain initialization.
"""
from global_utils import dump_results
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from exp.mixture_models.utils import get_num_true_clusters, get_log_labels
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
        data_manager = DataManager(data_config)
        logs = data_manager.get_tokenized_logs()
        true_assignments = data_manager.get_true_assignments()
        ev = Evaluator(true_assignments)

        n_true_clusters = get_num_true_clusters(true_assignments)
        log_labels = get_log_labels(true_assignments, n_labels)

        parser = MultinomialMixtureOnline(logs,
                                          n_true_clusters,
                                          is_classification=False,
                                          alpha=1.05,
                                          beta=1.05)

        parser.label_logs(log_labels, logs)

        clusters = parser.get_clusters(logs)
        results[data_config['name']] = ev.evaluate(clusters)

    return results


if __name__ == '__main__':
    n_labels = 50

    results = run_online_benchmark(n_labels)
    dump_results('online_benchmark.p', results)
