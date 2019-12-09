"""
Print and save the accuracies of a single Drain run on a target dataset (DATA_CONFIG) using random search.
"""
import matplotlib.pyplot as plt
from src.parsers.drain import Drain
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.utils import get_template_assignments
from src.parameter_searchers.parameter_random_searcher import ParameterRandomSearcher

N_RUNS = 30
DATA_CONFIG = DataConfigs.BGL

data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_log_entries()
true_assignments = get_template_assignments(DATA_CONFIG['assignments_path'])

parameter_ranges_dict = {
    'max_depth': (3, 8),
    'max_child': (20, 100),
    'sim_threshold': (0.1, 0.9),
}

parameter_searcher = ParameterRandomSearcher(Drain, parameter_ranges_dict, verbose=True, n_runs=N_RUNS)
parameter_searcher.search(tokenized_log_entries, true_assignments)

best_accuracy_history = parameter_searcher.best_accuracy_history
accuracy_indices = range(1, len(best_accuracy_history) + 1)

print('\nFinal Best Accuracy: {}'.format(best_accuracy_history[-1]))

plt.plot(accuracy_indices, best_accuracy_history)
plt.title('Drain Accuracy Over Number of Random Samples')
plt.ylabel('Accuracy')
plt.xlabel('Number of Samples')
plt.grid()
plt.show()
