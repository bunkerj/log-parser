from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager


def get_final_dataset_accuracies(Parser_class,
                                 data_configs,
                                 parameter_searcher=None,
                                 fixed_configs=None):
    final_accuracies = {}
    for data_config in data_configs:
        data_manager = DataManager(data_config)
        tokenized_logs = data_manager.get_tokenized_logs()
        true_assignments = data_manager.get_true_assignments()

        if parameter_searcher is not None:
            parameter_searcher.search(tokenized_logs, true_assignments)
            parameters = parameter_searcher.get_optimal_parameter_tuple()
        elif fixed_configs is not None:
            parameters = fixed_configs[data_config['name']]
        else:
            raise Exception('Invalid configuration setup')

        parser = Parser_class(tokenized_logs, *parameters)

        parser.parse()

        evaluator = Evaluator(true_assignments)
        accuracy = evaluator.evaluate(parser.cluster_templates)

        print('{}: Final {} Accuracy: {}'.format(parameters,
                                                 data_config['name'],
                                                 accuracy))
        final_accuracies[data_config['name']] = accuracy

    return final_accuracies


def update_average_list(current_average, new_list, n):
    for idx in range(len(current_average)):
        current_average[idx] += (new_list[idx] / n)
    return current_average
