"""
Print and save the accuracies of the listed datasets using Drain with random search.
"""
from exp.utils import get_final_dataset_accuracies, dump_results
from src.data_config import DataConfigs
from src.parameter_searchers.parameter_random_searcher import ParameterRandomSearcher
from src.parsers.drain import Drain

N_RUNS = 10

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
    'max_depth': (3, 8),
    'max_child': (20, 100),
    'sim_threshold': (0.1, 0.9),
}

parameter_searcher = ParameterRandomSearcher(Drain, parameter_ranges_dict, n_runs=N_RUNS)
final_best_accuracies = get_final_dataset_accuracies(Drain,
                                                     data_set_configs,
                                                     parameter_searcher=parameter_searcher)

dump_results('drain_dataset_comparison_random.p', final_best_accuracies)
