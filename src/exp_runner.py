from src.methods.iplom import Iplom
from src.constants import BGL_STRUCTURED_PATH
from src.helpers.evaluator import Evaluator
from src.helpers.parameter_grid_searcher import ParameterGridSearcher

parameter_ranges_dict = {
    'file_threshold': (0, 0.15, 0.05),
    'partition_threshold': (0, 0.35, 0.05),
    'lower_bound': (0.1, 0.35, 0.05),
    'upper_bound': (0.9, 1, 1),
    'goodness_threshold': (0.3, 0.65, 0.05)
}

parameter_grid_searcher = ParameterGridSearcher(BGL_STRUCTURED_PATH, parameter_ranges_dict)
parameter_grid_searcher.search()
parameter_grid_searcher.print_results()

iplom = Iplom(BGL_STRUCTURED_PATH, **parameter_grid_searcher.best_parameters_dict)
iplom.parse()
# iplom.print_cluster_templates()

evaluator = Evaluator(BGL_STRUCTURED_PATH, iplom.cluster_templates)
iplom_bgl_accuracy = evaluator.evaluate()

print('\nFinal IPLoM BGL Accuracy: {}'.format(iplom_bgl_accuracy))
