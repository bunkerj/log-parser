import numpy as np
from itertools import product
from src.parameter_searchers.parameter_search import ParameterSearch


class ParameterGridSearcher(ParameterSearch):
    def __init__(self, Parser_class, parameter_ranges_dict, verbose=False):
        super().__init__(Parser_class, parameter_ranges_dict, verbose=verbose)

    def _get_parameter_array_list(self):
        parameter_array_list = []
        for parameter_field in self.parameter_ranges_dict:
            parameter_range = self.parameter_ranges_dict[parameter_field]
            parameter_array_list.append(np.arange(*parameter_range))
        return parameter_array_list

    def _get_parameter_tuples(self):
        return list(product(*self._get_parameter_array_list()))
