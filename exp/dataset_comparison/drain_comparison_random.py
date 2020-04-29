"""
Print and save the accuracies of the listed datasets using Drain with random
search.
"""
from exp.utils import get_final_dataset_accuracies
from global_utils import dump_results
from src.data_config import DataConfigs
from src.parameter_searchers.parameter_random_searcher import \
    ParameterRandomSearcher
from src.parsers.enhanced_drain import EnhancedDrain


def run_drain_comparison_random(n_calls, data_set_configs,
                                parameter_ranges_dict, name):
    parameter_searcher = ParameterRandomSearcher(EnhancedDrain,
                                                 parameter_ranges_dict,
                                                 n_calls=n_calls)
    final_best_accuracies = \
        get_final_dataset_accuracies(EnhancedDrain,
                                     data_set_configs,
                                     parameter_searcher)

    dump_results(name, final_best_accuracies)


if __name__ == '__main__':
    n_calls = 5

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
        'max_depth': (3, 50),
        'max_child': (20, 100),
        'sim_threshold': (0.2, 0.8),
        'edit_ratio_threshold': (0, 0.5),
    }

    name = 'drain_enhanced_dataset_comparison_random.p'

    run_drain_comparison_random(n_calls, data_set_configs,
                                parameter_ranges_dict, name)
