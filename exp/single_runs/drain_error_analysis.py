"""
Analyze the errors from Drain for a given dataset.
"""
from src.parsers.drain import Drain
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager

DATA_CONFIG = DataConfigs.Proxifier

data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_log_entries()
true_assignments = data_manager.get_true_assignments()
parser = Drain(tokenized_log_entries, 8, 75, 0.4)

parser.parse()
evaluator = Evaluator(true_assignments)
evaluator.print_all_discrepancies(parser.cluster_templates)
