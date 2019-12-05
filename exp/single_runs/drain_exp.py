from src.parsers.drain import Drain
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from src.utils import read_csv

DATA_CONFIG = DataConfigs.BGL_FULL

data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_log_entries()
parser = Drain(tokenized_log_entries, 3, 100, 0.5)
parser.parse()

true_assignments = read_csv(DATA_CONFIG['assignments_path'])
evaluator = Evaluator(true_assignments, parser.cluster_templates)
accuracy = evaluator.evaluate()

print('Final Drain Accuracy: {}'.format(accuracy))
