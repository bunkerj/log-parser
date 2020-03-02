"""
Print and save the accuracies of the listed datasets using Drain with fixed
parameters.
"""
from exp.utils import get_final_dataset_accuracies
from utils import dump_results
from src.parsers.drain import Drain
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
                                 data_set_configs,
                                 fixed_configs=fixed_configs)

dump_results('drain_dataset_comparison.p', final_best_accuracies)
