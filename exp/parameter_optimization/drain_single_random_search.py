"""
Print and save the accuracies of a single Drain run on a target dataset
(DATA_CONFIG) using random search.
"""
from exp.utils import update_average_list
from utils import dump_results
from src.parsers.drain import Drain
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.utils import get_template_assignments
from src.parameter_searchers.parameter_random_searcher import \
    ParameterRandomSearcher

N_CALLS = 10
N_RUNS_FOR_AVERAGING = 5
DATA_CONFIG = DataConfigs.Proxifier

data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_log_entries()
true_assignments = get_template_assignments(DATA_CONFIG['assignments_path'])

parameter_ranges_dict = {
    'max_depth': (3, 8),
    'max_child': (20, 100),
    'sim_threshold': (0.1, 0.6),
}

average_best_accuracy_history = [0] * N_CALLS

for run in range(N_RUNS_FOR_AVERAGING):
    parameter_searcher = ParameterRandomSearcher(Drain, parameter_ranges_dict,
                                                 verbose=True, n_calls=N_CALLS)

    parameter_searcher.search(tokenized_log_entries, true_assignments)
    current_best_accuracy_history = parameter_searcher.best_accuracy_history

    average_best_accuracy_history = update_average_list(
        average_best_accuracy_history,
        current_best_accuracy_history,
        N_RUNS_FOR_AVERAGING)

dump_results('average_best_random_search_full_accuracy_history.p',
             average_best_accuracy_history)
