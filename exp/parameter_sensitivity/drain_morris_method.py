"""
Generate the time taken in minutes and accuracies for corresponding sampled
Drain configurations.

The goal is to fill the "morris_data" dictionary with: timings, accuracies,
parameter names, parameter configurations, and the Morris sensitivity indices
for the timings and accuracies.
"""
import numpy as np
from time import time
from global_utils import dump_results
from src.parsers.drain import Drain
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from src.utils import get_template_assignments
from SALib.analyze.morris import analyze
from SALib.sample.morris import sample

NUM_LEVELS = 4
CONF_LEVEL = 0.95
N_TRAJECTORIES = 100
DATA_CONFIG = DataConfigs.BGL_FULL

parameter_ranges_dict = {
    'Max Depth': (3, 8),
    'Max Child': (20, 100),
    'Sim Threshold': (0.1, 0.9),
}

data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_log_entries()
true_assignments = get_template_assignments(DATA_CONFIG['assignments_path'])
evaluator = Evaluator(true_assignments)

problem = {
    'num_vars': len(parameter_ranges_dict),
    'names': list(parameter_ranges_dict.keys()),
    'bounds': [parameter_ranges_dict[k] for k in parameter_ranges_dict],
}

morris_data = {
    'timing': [],
    'accuracy': [],
    'parameter_names': list(parameter_ranges_dict.keys()),
    'parameters': sample(problem, N_TRAJECTORIES, num_levels=NUM_LEVELS),
    'accuracy_sens_indices': None,
    'timing_sens_indices': None,
}

for idx, parameter_tuple in enumerate(morris_data['parameters']):
    print('Run {}/{}'.format(idx + 1, len(morris_data['parameters'])))

    parser = Drain(tokenized_log_entries, *parameter_tuple)

    start_time = time()
    parser.parse()
    minutes_to_parse = (time() - start_time) / 60

    accuracy = evaluator.evaluate(parser.cluster_templates)

    morris_data['timing'].append(minutes_to_parse)
    morris_data['accuracy'].append(accuracy)

morris_data['accuracy_sens_indices'] = \
    analyze(problem,
            morris_data['parameters'],
            np.array(morris_data['accuracy']),
            conf_level=CONF_LEVEL,
            num_levels=NUM_LEVELS)

morris_data['timing_sens_indices'] = \
    analyze(problem,
            morris_data['parameters'],
            np.array(morris_data['timing']),
            conf_level=CONF_LEVEL,
            num_levels=NUM_LEVELS)

dump_results('drain_morris_data.p', morris_data)
