from src.methods.iplom import Iplom
from src.data_config import DataConfig
from src.helpers.evaluator import Evaluator
from src.helpers.parameter_grid_searcher import ParameterGridSearcher

# parameter_ranges_dict = {
#     'file_threshold': (0, 0.15, 0.05),
#     'partition_threshold': (0, 0.35, 0.05),
#     'lower_bound': (0.1, 0.35, 0.05),
#     'upper_bound': (0.9, 1, 1),
#     'goodness_threshold': (0.3, 0.65, 0.05)
# }
#
# parameter_grid_searcher = ParameterGridSearcher(DataConfig.BGL, parameter_ranges_dict)
# parameter_grid_searcher.search()
# parameter_grid_searcher.print_results()

iplom = Iplom(DataConfig.BGL, **{
    'file_threshold': 0.0,
    'partition_threshold': 0.0,
    'lower_bound': 0.1,
    'upper_bound': 0.9,
    'goodness_threshold': 0.35,
})
iplom.parse()
# iplom.print_cluster_templates()

evaluator = Evaluator(DataConfig.BGL, iplom.cluster_templates)
iplom_bgl_accuracy = evaluator.evaluate()

print('\nFinal IPLoM BGL Accuracy: {}'.format(iplom_bgl_accuracy))
