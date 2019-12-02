import os
import pickle
from constants import RESULTS_DIR
from src.helpers.evaluator import Evaluator
from src.helpers.parameter_grid_searcher import ParameterGridSearcher


def get_final_dataset_accuracies(Parser_class,
                                 data_set_configs,
                                 parameter_ranges_dict=None,
                                 fixed_configs=None):
    final_best_accuracies = {}
    for data_set_config in data_set_configs:
        if parameter_ranges_dict is not None:
            parameter_grid_searcher = ParameterGridSearcher(Parser_class,
                                                            data_set_config,
                                                            parameter_ranges_dict)
            parameter_grid_searcher.search()
            parser = Parser_class(data_set_config,
                                  **parameter_grid_searcher.best_parameters_dict)
        elif fixed_configs is not None:
            parser = Parser_class(data_set_config,
                                  *fixed_configs[data_set_config['name']])
        else:
            raise Exception('Invalid configuration setup')

        parser.parse()

        evaluator = Evaluator(data_set_config, parser.cluster_templates)
        accuracy = evaluator.evaluate()

        print('Final {} Accuracy: {}'.format(data_set_config['name'], accuracy))
        final_best_accuracies[data_set_config['name']] = accuracy
    return final_best_accuracies


def dump_results(name, results):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    path = os.path.join(RESULTS_DIR, name)
    pickle.dump(results, open(path, 'wb'))
