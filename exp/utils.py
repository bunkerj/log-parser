import os
import pickle
from constants import RESULTS_DIR
from src.utils import get_template_assignments
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager


def get_final_dataset_accuracies(Parser_class,
                                 data_set_configs,
                                 parameter_searcher=None,
                                 fixed_configs=None):
    final_accuracies = {}
    for data_set_config in data_set_configs:
        data_manager = DataManager(data_set_config)
        tokenized_log_entries = data_manager.get_tokenized_log_entries()
        true_assignments = get_template_assignments(
            data_set_config['assignments_path'])

        if parameter_searcher is not None:
            parameter_searcher.search(tokenized_log_entries, true_assignments)
            parameters = parameter_searcher.get_optimal_parameter_tuple()
        elif fixed_configs is not None:
            parameters = fixed_configs[data_set_config['name']]
        else:
            raise Exception('Invalid configuration setup')

        parser = Parser_class(tokenized_log_entries, *parameters)

        parser.parse()

        evaluator = Evaluator(true_assignments)
        accuracy = evaluator.evaluate(parser.cluster_templates)

        print('{}: Final {} Accuracy: {}'.format(parameters,
                                                 data_set_config['name'],
                                                 accuracy))
        final_accuracies[data_set_config['name']] = accuracy

    return final_accuracies


def dump_results(name, results):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    path = os.path.join(RESULTS_DIR, name)
    pickle.dump(results, open(path, 'wb'))


def update_average_list(current_average, new_list, n):
    for idx in range(len(current_average)):
        current_average[idx] += (new_list[idx] / n)
    return current_average
