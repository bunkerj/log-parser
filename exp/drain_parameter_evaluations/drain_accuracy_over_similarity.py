"""
Plot the accuracy over the similarity measure.
"""
import numpy as np
from global_utils import dump_results
from src.parsers.drain import Drain
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager


def run_drain_accuracy_breakdown_over_similarity(data_config,
                                                 sim_thresholds,
                                                 name):
    accuracies = []
    type1_error_ratios = []
    type2_error_ratios = []

    data_manager = DataManager(data_config)
    tokenized_log_entries = data_manager.get_tokenized_log_entries()
    true_assignments = data_manager.get_true_assignments()
    evaluator = Evaluator(true_assignments)

    for sim_threshold in sim_thresholds:
        parser = Drain(tokenized_log_entries, 8, 75, sim_threshold)
        parser.parse()
        cluster_templates = parser.cluster_templates

        accuracy = evaluator.evaluate(cluster_templates)
        type1_error_ratio = evaluator.get_type1_error_ratio()
        type2_error_ratio = evaluator.get_type2_error_ratio()

        type1_error_ratios.append(type1_error_ratio)
        type2_error_ratios.append(type2_error_ratio)
        accuracies.append(accuracy)

        results = {
            'sim_thresholds': sim_thresholds,
            'accuracies': accuracies,
            'type1_error_ratios': type1_error_ratios,
            'type2_error_ratios': type2_error_ratios,
        }

        dump_results(name, results)


if __name__ == '__main__':
    data_config = DataConfigs.Proxifier
    sim_thresholds = np.arange(0.01, 1.0, 0.01)
    name = 'drain_accuracy_over_similarity.p'

    run_drain_accuracy_breakdown_over_similarity(data_config,
                                                 sim_thresholds,
                                                 name)
