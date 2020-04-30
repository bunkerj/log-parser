"""
Find the best Drain parameters on a chosen dataset.
"""
from global_utils import dump_results
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.parameter_searchers.parameter_random_searcher import \
    ParameterRandomSearcher
from src.parsers.enhanced_drain import EnhancedDrain


def run_drain_best_parameters(n_calls, data_config, parameter_ranges_dict,
                              name):
    data_manager = DataManager(data_config)
    tokenized_log_entries = data_manager.get_tokenized_log_entries()
    true_assignments = data_manager.get_true_assignments()

    parameter_searcher = ParameterRandomSearcher(EnhancedDrain,
                                                 parameter_ranges_dict,
                                                 n_calls=n_calls, verbose=True)
    parameter_searcher.search(tokenized_log_entries, true_assignments)
    parameters = tuple(parameter_searcher.get_optimal_parameter_tuple())

    dump_results(name, parameters)


if __name__ == '__main__':
    n_calls = 100
    data_config = DataConfigs.Apache
    parameter_ranges_dict = {
        'max_depth': (3, 50),
        'max_child': (20, 100),
        'sim_threshold': (0.1, 0.8),
        'edit_ratio_threshold': (0, 1),
    }
    name = 'drain_best_parameters.p'

    run_drain_best_parameters(n_calls, data_config, parameter_ranges_dict, name)
