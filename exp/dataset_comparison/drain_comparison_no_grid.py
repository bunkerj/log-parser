from exp.utils import get_final_dataset_accuracies, dump_results
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
    'Android': (6, 100, 0.2),
    'Apache': (4, 100, 0.5),
    'BGL': (4, 100, 0.5),
    'Hadoop': (4, 100, 0.5),
    'HDFS': (4, 100, 0.5),
    'HealthApp': (4, 100, 0.2),
    'HPC': (4, 100, 0.5),
    'Linux': (6, 100, 0.39),
    'Mac': (6, 100, 0.7),
    'OpenSSH': (5, 100, 0.6),
    'OpenStack': (5, 100, 0.5),
    'Proxifier': (3, 100, 0.6),
    'Spark': (4, 100, 0.5),
    'Thunderbird': (4, 100, 0.5),
    'Windows': (5, 100, 0.7),
    'Zookeeper': (4, 100, 0.5),
}

final_best_accuracies = get_final_dataset_accuracies(Drain,
                                                     data_set_configs,
                                                     fixed_configs=fixed_configs)

dump_results('drain_dataset_comparison_no_grid.p', final_best_accuracies)
