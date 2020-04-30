"""
Print and save the accuracies of the listed datasets using Drain with grid
search.
"""
from exp.utils import get_final_dataset_accuracies
from global_utils import dump_results
from src.parameter_searchers.parameter_grid_searcher import \
    ParameterGridSearcher
from src.parsers.drain import Drain
from src.data_config import DataConfigs


def run_drain_comparison_grid(data_set_configs, parameter_ranges_dict):
    parameter_searcher = ParameterGridSearcher(Drain, parameter_ranges_dict)
    final_best_accuracies = \
        get_final_dataset_accuracies(Drain,
                                     data_set_configs,
                                     parameter_searcher=parameter_searcher)
    return final_best_accuracies


if __name__ == '__main__':
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
        'max_depth': (3, 8, 1),
        'max_child': (100, 101, 100),
        'sim_threshold': (0.1, 0.9, 0.05),
    }

    results = run_drain_comparison_grid(data_set_configs,
                                        parameter_ranges_dict)
    dump_results('drain_comparison_grid.p', results)
