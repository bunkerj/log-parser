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
from SALib.analyze.morris import analyze
from SALib.sample.morris import sample


def run_drain_morris_method(num_levels, conf_level, n_trajectories, data_config,
                            parameter_ranges_dict, name):
    data_manager = DataManager(data_config)
    tokenized_log_entries = data_manager.get_tokenized_log_entries()
    true_assignments = data_manager.get_true_assignments()
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
        'parameters': sample(problem, n_trajectories, num_levels=num_levels),
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
                conf_level=conf_level,
                num_levels=num_levels)

    morris_data['timing_sens_indices'] = \
        analyze(problem,
                morris_data['parameters'],
                np.array(morris_data['timing']),
                conf_level=conf_level,
                num_levels=num_levels)

    dump_results(name, morris_data)


if __name__ == '__main__':
    num_levels = 4
    conf_level = 0.95
    n_trajectories = 100
    data_config = DataConfigs.Apache
    parameter_ranges_dict = {
        'Max Depth': (3, 8),
        'Max Child': (20, 100),
        'Sim Threshold': (0.1, 0.9),
    }
    name = 'drain_morris_data.p'

    run_drain_morris_method(num_levels, conf_level, n_trajectories, data_config,
                            parameter_ranges_dict, name)
