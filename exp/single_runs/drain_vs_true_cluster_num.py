from global_utils import dump_results, get_num_true_clusters
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.parsers.drain import Drain


def run_drain_vs_true_cluster_num(data_configs, drain_parameters):
    results = {}
    for data_config in data_configs:
        name = data_config['name']
        print('{}...'.format(name))
        results[name] = {}

        data_manager = DataManager(data_config)
        logs = data_manager.get_tokenized_logs()
        true_assignments = data_manager.get_true_assignments()
        num_true_clusters = get_num_true_clusters(true_assignments)

        drain = Drain(logs, *drain_parameters[name])
        drain.parse()
        drain_clusters = drain.cluster_templates
        num_drain_clusters = len(drain_clusters)

        results[name] = {'Drain': num_drain_clusters,
                         'Truth': num_true_clusters}

    return results


if __name__ == '__main__':
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

    drain_parameters = {
        'Android': (5, 100, 0.21),
        'Apache': (11, 100, 0.76),
        'BGL': (5, 100, 0.54),
        'Hadoop': (3, 100, 0.66),
        'HDFS': (3, 100, 0.48),
        'HealthApp': (3, 100, 0.30),
        'HPC': (3, 100, 0.22),
        'Linux': (4, 100, 0.40),
        'Mac': (4, 100, 0.80),
        'OpenSSH': (4, 100, 0.71),
        'OpenStack': (3, 100, 0.80),
        'Proxifier': (50, 100, 0.62),
        'Spark': (6, 100, 0.75),
        'Thunderbird': (4, 100, 0.70),
        'Windows': (7, 100, 0.42),
        'Zookeeper': (4, 100, 0.60),
    }

    results = run_drain_vs_true_cluster_num(data_configs, drain_parameters)
    dump_results('drain_vs_true_cluster_num.p', results)
