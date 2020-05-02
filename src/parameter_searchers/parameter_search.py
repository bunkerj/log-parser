from abc import ABC, abstractmethod
from src.helpers.evaluator import Evaluator


class ParameterSearch(ABC):
    def __init__(self, Parser_class, parameter_ranges_dict, verbose=False):
        self.best_accuracy = -1
        self.best_parameters_dict = {}
        self.best_accuracy_history = []
        self.parameter_ranges_dict = parameter_ranges_dict
        self.verbose = verbose
        self.Parser_class = Parser_class

    def search(self, tokenized_logs, true_assignments):
        self._reset_dynamic_fields()
        current_iteration = 1
        evaluator = Evaluator(true_assignments)
        parameter_tuples = self._get_parameter_tuples()
        total_iterations = len(parameter_tuples)
        for parameter_tuple in parameter_tuples:
            parameter_dict = self._get_parameter_dict(parameter_tuple)
            parser = self.Parser_class(tokenized_logs, **parameter_dict)
            parser.parse()

            current_accuracy = evaluator.evaluate(parser.cluster_templates)

            if current_accuracy > self.best_accuracy:
                self.best_parameters_dict = parameter_dict
                self.best_accuracy = current_accuracy

            if self.verbose:
                self._print_status(current_accuracy, current_iteration,
                                   parameter_tuple, total_iterations)

            self.best_accuracy_history.append(self.best_accuracy)
            current_iteration += 1

    def get_optimal_parameter_tuple(self):
        return tuple(self.best_parameters_dict.values())

    def _print_status(self, current_accuracy, current_iteration,
                      parameter_tuple, total_iterations):
        msg = '{}/{} -- Best Acc: {} -- Current Acc: {} -- Current Params: {}'
        filled_msg = msg.format(current_iteration,
                                total_iterations,
                                self.best_accuracy,
                                current_accuracy,
                                parameter_tuple)
        print(filled_msg)

    def _reset_dynamic_fields(self):
        self.best_accuracy = -1
        self.best_parameters_dict = {}
        self.best_accuracy_history = []

    def print_results(self):
        print('\n---- Best Parameters (Accuracy: {}) ----'.format(
            self.best_accuracy))
        for name in self.best_parameters_dict:
            value = self.best_parameters_dict[name]
            print('{}: {}'.format(name, value))

    def _get_parameter_dict(self, parameter_tuple):
        return {parameter_name: parameter_tuple[idx] for idx, parameter_name in
                enumerate(self.parameter_ranges_dict)}

    @abstractmethod
    def _get_parameter_tuples(self):
        """
        Returns list of tuples where each tuple represents a parameter
        configuration.
        """
        pass
