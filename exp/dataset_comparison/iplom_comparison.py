"""
Print and save the accuracies of the listed datasets using IPLoM with fixed
parameters.
"""
from exp.utils import get_final_dataset_accuracies, dump_results
from src.parsers.iplom import Iplom
from src.data_config import DataConfigs

data_set_configs = [
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

fixed_configs = {
    'Android': (0, 0, 0.3, 0.9, 0.25),
    'Apache': (0, 0, 0.4, 0.9, 0.3),
    'BGL': (0, 0, 0.01, 0.9, 0.4),
    'Hadoop': (0, 0, 0.2, 0.9, 0.4),
    'HDFS': (0, 0, 0.25, 0.9, 0.35),
    'HealthApp': (0, 0, 0.3, 0.9, 0.25),
    'HPC': (0, 0, 0.25, 0.9, 0.58),
    'Linux': (0, 0, 0.3, 0.9, 0.3),
    'Mac': (0, 0, 0.25, 0.9, 0.3),
    'OpenSSH': (0, 0, 0.25, 0.9, 0.78),
    'OpenStack': (0, 0, 0.25, 0.9, 0.9),
    'Proxifier': (0, 0, 0.25, 0.9, 0.9),
    'Spark': (0, 0, 0.3, 0.9, 0.35),
    'Thunderbird': (0, 0, 0.2, 0.9, 0.3),
    'Windows': (0, 0, 0.25, 0.9, 0.3),
    'Zookeeper': (0, 0, 0.7, 0.9, 0.4),
}

final_best_accuracies = get_final_dataset_accuracies(Iplom,
                                                     data_set_configs,
                                                     fixed_configs=fixed_configs)

dump_results('iplom_dataset_comparison.p', final_best_accuracies)
