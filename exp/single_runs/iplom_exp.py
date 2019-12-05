from src.parsers.iplom import Iplom
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator

DATA_CONFIG = DataConfigs.OpenStack

parser = Iplom(DATA_CONFIG, **{
    'file_threshold': 0.0,
    'partition_threshold': 0.0,
    'lower_bound': 0.1,
    'upper_bound': 0.9,
    'goodness_threshold': 0.35,
})
parser.parse()
parser.print_cluster_templates()

evaluator = Evaluator(DATA_CONFIG, parser.cluster_templates)
accuracy = evaluator.evaluate()

print('Final IPLoM Accuracy: {}'.format(accuracy))
