from src.parsers.iplom import Iplom
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator

iplom = Iplom(DataConfigs.OpenStack, **{
    'file_threshold': 0.0,
    'partition_threshold': 0.0,
    'lower_bound': 0.1,
    'upper_bound': 0.9,
    'goodness_threshold': 0.35,
})
iplom.parse()
iplom.print_cluster_templates()

evaluator = Evaluator(DataConfigs.OpenStack, iplom.cluster_templates)
iplom_bgl_accuracy = evaluator.evaluate()

print('Final IPLoM BGL Accuracy: {}'.format(iplom_bgl_accuracy))
