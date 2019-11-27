import os
import pickle
from src.helpers.evaluator import Evaluator
from src.helpers.parameter_grid_searcher import ParameterGridSearcher

RESULTS_DIR = '../results'


def get_final_dataset_accuracies(Parser_class, data_set_configs, parameter_ranges_dict):
    final_best_accuracies = {}
    for data_set_config in data_set_configs:
        parameter_grid_searcher = ParameterGridSearcher(data_set_config, parameter_ranges_dict)
        parameter_grid_searcher.search()

        parser = Parser_class(data_set_config, **parameter_grid_searcher.best_parameters_dict)
        parser.parse()

        evaluator = Evaluator(data_set_config, parser.cluster_templates)
        iplom_accuracy = evaluator.evaluate()

        print('Final IPLoM {} Accuracy: {}'.format(data_set_config['name'], iplom_accuracy))
        final_best_accuracies[data_set_config['name']] = iplom_accuracy
    return final_best_accuracies


def dump_results(name, results):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    path = os.path.join('../results/', name)
    pickle.dump(results, open(path, 'wb'))
