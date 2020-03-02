"""
Find the best Drain parameters on a chosen dataset.
"""
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.parameter_searchers.parameter_random_searcher import \
    ParameterRandomSearcher
from src.parsers.enhanced_drain import EnhancedDrain

N_CALLS = 250
DATA_CONFIG = DataConfigs.Linux

parameter_ranges_dict = {
    'max_depth': (3, 50),
    'max_child': (20, 100),
    'sim_threshold': (0.1, 0.9),
    'edit_ratio_threshold': (0, 1),
}

data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_log_entries()
true_assignments = data_manager.get_true_assignments()

parameter_searcher = ParameterRandomSearcher(EnhancedDrain,
                                             parameter_ranges_dict,
                                             n_calls=N_CALLS, verbose=True)
parameter_searcher.search(tokenized_log_entries, true_assignments)
parameters = tuple(parameter_searcher.get_optimal_parameter_tuple())

print('\nBest parameters: {}'.format(parameters))
