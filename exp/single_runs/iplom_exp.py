from src.helpers.data_manager import DataManager
from src.parsers.iplom import Iplom
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.utils import read_csv

DATA_CONFIG = DataConfigs.OpenStack

data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_log_entries()
parser = Iplom(tokenized_log_entries, **{
    'file_threshold': 0.0,
    'partition_threshold': 0.0,
    'lower_bound': 0.1,
    'upper_bound': 0.9,
    'goodness_threshold': 0.35,
})
parser.parse()
parser.print_cluster_templates()

true_assignments = read_csv(DATA_CONFIG['assignments_path'])
evaluator = Evaluator(true_assignments, parser.cluster_templates)
accuracy = evaluator.evaluate()

print('Final IPLoM Accuracy: {}'.format(accuracy))