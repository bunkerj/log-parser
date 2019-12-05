from src.parsers.drain import Drain
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator

DATA_CONFIG = DataConfigs.BGL_FULL

parser = Drain(DATA_CONFIG, 3, 100, 0.5)
parser.parse()

evaluator = Evaluator(DATA_CONFIG, parser.cluster_templates)
accuracy = evaluator.evaluate()

print('Final Drain Accuracy: {}'.format(accuracy))
