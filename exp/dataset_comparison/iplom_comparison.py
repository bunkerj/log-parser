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

parameter_ranges_dict = {
    'file_threshold': (0, 0.15, 0.05),
    'partition_threshold': (0, 0.35, 0.05),
    'lower_bound': (0.1, 0.35, 0.05),
    'upper_bound': (0.9, 1, 1),
    'goodness_threshold': (0.3, 1, 0.1)
}

final_best_accuracies = get_final_dataset_accuracies(Iplom,
                                                     data_set_configs,
                                                     parameter_ranges_dict=parameter_ranges_dict)

dump_results('iplom_dataset_comparison.p', final_best_accuracies)
