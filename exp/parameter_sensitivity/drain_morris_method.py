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


def run_drain_morris_method(data_config, num_levels, conf_level, n_trajectories,
                            parameter_ranges_dict):
    data_manager = DataManager(data_config)
    tokenized_logs = data_manager.get_tokenized_logs()
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

        parser = Drain(tokenized_logs, *parameter_tuple)

        start_time = time()
        parser.parse()
        minutes_to_parse = (time() - start_time) / 60

        accuracy = evaluator.get_accuracy(parser.cluster_templates)

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

    return morris_data


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

    results = run_drain_morris_method(data_config, num_levels, conf_level,
                                      n_trajectories, parameter_ranges_dict)
    dump_results('drain_morris_method.p', results)
