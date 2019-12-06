import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from src.utils import read_csv
from src.parsers.drain import Drain
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager

DATA_CONFIG = DataConfigs.BGL

base_config = {
    'max_depth': 3,
    'max_child': 100,
    'sim_threshold': 0.5
}

parameter_ranges_dict = {
    'max_depth': (2, 22, 2),
    'max_child': (10, 110, 10),
    'sim_threshold': (0.1, 0.90, 0.08),
}

accuracies = {}
tree_depths = list(range(3, 30, 1))
true_assignments = read_csv(DATA_CONFIG['assignments_path'])

data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_log_entries()

for parameter_field in parameter_ranges_dict:
    parameter_range = parameter_ranges_dict[parameter_field]
    for parameter_value in np.arange(*parameter_range):
        base_config_copy = copy(base_config)
        base_config_copy[parameter_field] = parameter_value

        parser = Drain(tokenized_log_entries, **base_config_copy)
        parser.parse()

        evaluator = Evaluator(true_assignments, parser.cluster_templates)
        if parameter_field not in accuracies:
            accuracies[parameter_field] = []
        accuracies[parameter_field].append(evaluator.evaluate())

boxplot_data = [
    accuracies['max_depth'],
    accuracies['max_child'],
    accuracies['sim_threshold'],
]

plt.boxplot(boxplot_data, sym='')
plt.title('Drain Parameter Sensitivity')
plt.ylabel('Percentage Accuracy')
plt.xticks(range(1, 4), ['max_depth', 'max_child', 'sim_threshold'])
plt.grid()
plt.show()
