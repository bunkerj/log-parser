import os
import pickle
from constants import RESULTS_DIR
from src.utils import read_csv
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from src.helpers.parameter_grid_searcher import ParameterGridSearcher


def get_final_dataset_accuracies(Parser_class,
                                 data_set_configs,
                                 parameter_ranges_dict=None,
                                 fixed_configs=None):
    final_best_accuracies = {}
    for data_set_config in data_set_configs:
        data_manager = DataManager(data_set_config)
        tokenized_log_entries = data_manager.get_tokenized_log_entries()
        true_assignments = read_csv(data_set_config['assignments_path'])

        if parameter_ranges_dict is not None:
            parameter_grid_searcher = ParameterGridSearcher(Parser_class,
                                                            parameter_ranges_dict)
            parameter_grid_searcher.search(tokenized_log_entries, true_assignments)
            parser = Parser_class(tokenized_log_entries,
                                  **parameter_grid_searcher.best_parameters_dict)
        elif fixed_configs is not None:
            parser = Parser_class(tokenized_log_entries,
                                  *fixed_configs[data_set_config['name']])
        else:
            raise Exception('Invalid configuration setup')

        parser.parse()

        true_assignments = read_csv(data_set_config['assignments_path'])
        evaluator = Evaluator(true_assignments, parser.cluster_templates)
        accuracy = evaluator.evaluate()

        print('Final {} Accuracy: {}'.format(data_set_config['name'], accuracy))
        final_best_accuracies[data_set_config['name']] = accuracy
    return final_best_accuracies


def dump_results(name, results):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    path = os.path.join(RESULTS_DIR, name)
    pickle.dump(results, open(path, 'wb'))
