import numpy as np
from random import uniform
from itertools import product
from src.parameter_searchers.parameter_search import ParameterSearch


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
            parameter_tuple = tuple(uniform(*self.parameter_ranges_dict[parameter_field])
                                    for parameter_field in self.parameter_ranges_dict)
            parameter_tuples.append(parameter_tuple)
        return parameter_tuples
