"""
Plot boxplots for different accuracy values resulting from different parameter
settings.
"""
import numpy as np
from copy import copy
from global_utils import dump_results
from src.parsers.drain import Drain
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager


def run_drain_parameter_sensitivity(data_config,
                                    base_config,
                                    parameter_ranges_dict):
    accuracies = {}
    data_manager = DataManager(data_config)
    tokenized_log_entries = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()
    evaluator = Evaluator(true_assignments)

    for parameter_field in parameter_ranges_dict:
        parameter_range = parameter_ranges_dict[parameter_field]
        for parameter_value in np.arange(*parameter_range):
            base_config_copy = copy(base_config)
            base_config_copy[parameter_field] = parameter_value

            parser = Drain(tokenized_log_entries, **base_config_copy)
            parser.parse()

            if parameter_field not in accuracies:
                accuracies[parameter_field] = []
            accuracies[parameter_field].append(
                evaluator.evaluate(parser.cluster_templates))

    return [
        accuracies['max_depth'],
        accuracies['max_child'],
        accuracies['sim_threshold'],
    ]


if __name__ == '__main__':
    data_config = DataConfigs.BGL
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

    results = run_drain_parameter_sensitivity(data_config, base_config,
                                              parameter_ranges_dict)
    dump_results('drain_parameter_sensitivity.p', results)
