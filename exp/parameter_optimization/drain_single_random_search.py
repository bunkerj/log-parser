"""
Print and save the accuracies of a single Drain run on a target dataset
(DATA_CONFIG) using random search.
"""
from exp.utils import update_average_list
from global_utils import dump_results
from src.parsers.drain import Drain
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.parameter_searchers.parameter_random_searcher import \
    ParameterRandomSearcher


def run_drain_single_random_search(n_calls, n_runs, data_config, name):
    data_manager = DataManager(data_config)
    tokenized_log_entries = data_manager.get_tokenized_log_entries()
    true_assignments = data_manager.get_true_assignments()
    average_best_accuracy_history = [0] * n_calls

    for run in range(n_runs):
        parameter_searcher = ParameterRandomSearcher(Drain,
                                                     parameter_ranges_dict,
                                                     verbose=True,
                                                     n_calls=n_calls)

        parameter_searcher.search(tokenized_log_entries, true_assignments)
        current_best_accuracy_history = parameter_searcher.best_accuracy_history

        average_best_accuracy_history = update_average_list(
            average_best_accuracy_history,
            current_best_accuracy_history,
            n_runs)

    dump_results(name, average_best_accuracy_history)


if __name__ == '__main__':
    n_calls = 10
    n_runs = 5
    data_config = DataConfigs.Proxifier
    parameter_ranges_dict = {
        'max_depth': (3, 8),
        'max_child': (20, 100),
        'sim_threshold': (0.1, 0.6),
    }
    name = 'drain_single_random_search.p'

    run_drain_single_random_search(n_calls, n_runs, data_config, name)
