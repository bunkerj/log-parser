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
        accuracy = evaluator.get_accuracy(parser.cluster_templates)

        print('{}: Final {} Accuracy: {}'.format(parameters,
                                                 data_config['name'],
                                                 accuracy))
        final_accuracies[data_config['name']] = accuracy

    return final_accuracies


def update_average_list(current_average, new_list, n):
    for idx in range(len(current_average)):
        current_average[idx] += (new_list[idx] / n)
    return current_average


def get_trivial_coreset(logs):
    logs_cs = []
    indices_cs = []
    log_str_counts = {}

    for idx, log in enumerate(logs):
        log_str = ' '.join(log)
        if log_str not in log_str_counts:
            logs_cs.append(log)
            indices_cs.append(idx)
            log_str_counts[log_str] = 0
        log_str_counts[log_str] += 1

    w_cs = []
    for log in logs_cs:
        log_str = ' '.join(log)
        count = log_str_counts[log_str]
        w_cs.append(count)

    return w_cs, logs_cs, indices_cs


def get_extended_cs(w_cs, logs_cs, indices_cs):
    w_cs_ext = []
    logs_cs_ext = []
    indices_cs_ext = []
    for idx in range(len(w_cs)):
        count = int(w_cs[idx])
        log_cs = logs_cs[idx]
        idx_cs = indices_cs[idx]
        w_cs_ext.extend([1 for _ in range(count)])
        logs_cs_ext.extend([log_cs for _ in range(count)])
        indices_cs_ext.extend([idx_cs for _ in range(count)])
    return w_cs_ext, logs_cs_ext, indices_cs_ext
