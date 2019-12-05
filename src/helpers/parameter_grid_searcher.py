import numpy as np
from itertools import product
from src.helpers.evaluator import Evaluator


class ParameterGridSearcher:
    def __init__(self, Parser_class, parameter_ranges_dict, verbose=False):
        self.parameter_ranges_dict = parameter_ranges_dict
        self.best_accuracy = -1
        self.best_parameters_dict = {}
        self.verbose = verbose
        self.Parser_class = Parser_class

    def search(self, tokenized_log_entries, true_assignments):
        current_iteration = 1
        total_iterations = self._get_total_iterations_count()
        for parameter_tuple in self._get_parameter_combinations():
            parameter_dict = self._get_parameter_dict(parameter_tuple)
            parser = self.Parser_class(tokenized_log_entries, **parameter_dict)
            parser.parse()

            evaluator = Evaluator(true_assignments, parser.cluster_templates)
            current_accuracy = evaluator.evaluate()

            if current_accuracy > self.best_accuracy:
                self.best_parameters_dict = parameter_dict
                self.best_accuracy = current_accuracy

            if self.verbose:
                msg = '{}/{} ---- Current Accuracy: {} ---- Best Accuracy: {}'
                print(msg.format(current_iteration, total_iterations, current_accuracy, self.best_accuracy))

            current_iteration += 1

    def print_results(self):
        print('\n---- Best Parameters (Accuracy: {}) ----'.format(self.best_accuracy))
        for name in self.best_parameters_dict:
            value = self.best_parameters_dict[name]
            print('{}: {}'.format(name, value))

    def _get_parameter_dict(self, parameter_tuple):
        return {parameter_name: parameter_tuple[idx] for idx, parameter_name in
                enumerate(self.parameter_ranges_dict)}

    def _get_parameter_array_list(self):
        parameter_array_list = []
        for parameter_field in self.parameter_ranges_dict:
            parameter_range = self.parameter_ranges_dict[parameter_field]
            parameter_array_list.append(np.arange(*parameter_range))
        return parameter_array_list

    def _get_parameter_combinations(self):
        return product(*self._get_parameter_array_list())

    def _get_total_iterations_count(self):
        return np.prod([len(arr) for arr in self._get_parameter_array_list()])
