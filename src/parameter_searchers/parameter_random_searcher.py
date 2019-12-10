import numpy as np
from src.parameter_searchers.parameter_search import ParameterSearch
from src.utils import get_random_parameter_tuple


class ParameterRandomSearcher(ParameterSearch):
    def __init__(self, Parser_class, parameter_ranges_dict, verbose=False, n_runs=10):
        super().__init__(Parser_class, parameter_ranges_dict, verbose=verbose)
        self.n_runs = n_runs

    def _get_parameter_array_list(self):
        parameter_array_list = []
        for parameter_field in self.parameter_ranges_dict:
            parameter_range = self.parameter_ranges_dict[parameter_field]
            parameter_array_list.append(np.arange(*parameter_range))
        return parameter_array_list

    def _get_parameter_tuples(self):
        parameter_tuples = []
        for run in range(self.n_runs):
            parameter_tuple = get_random_parameter_tuple(self.parameter_ranges_dict)
            parameter_tuples.append(parameter_tuple)
        return parameter_tuples
