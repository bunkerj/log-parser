"""
Print and save the accuracies of the listed datasets using Drain with fixed
parameters.
"""
from exp.utils import get_final_dataset_accuracies
from global_utils import dump_results
from src.parsers.drain import Drain
from src.data_config import DataConfigs


def run_drain_comparison(data_configs):
    fixed_configs = {
        'Android': (5, 100, 0.2),
        'Apache': (3, 100, 0.5),
        'BGL': (3, 100, 0.5),
        'Hadoop': (3, 100, 0.5),
        'HDFS': (3, 100, 0.5),
        'HealthApp': (3, 100, 0.2),
        'HPC': (3, 100, 0.5),
        'Linux': (5, 100, 0.39),
        'Mac': (5, 100, 0.7),
        'OpenSSH': (4, 100, 0.6),
        'OpenStack': (4, 100, 0.5),
        'Proxifier': (2, 100, 0.6),
        'Spark': (3, 100, 0.5),
        'Thunderbird': (3, 100, 0.5),
        'Windows': (4, 100, 0.7),
        'Zookeeper': (3, 100, 0.5),
    }

    final_best_accuracies = \
        get_final_dataset_accuracies(Drain,
                                     data_configs,
                                     fixed_configs=fixed_configs)
    return final_best_accuracies


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

    results = run_drain_comparison(data_configs)
    dump_results('drain_comparison.p', results)
