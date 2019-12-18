"""
Analyze the errors from Drain for a given dataset.
"""
from src.parsers.drain import Drain
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from src.utils import get_template_assignments

DATA_CONFIG = DataConfigs.BGL

data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_log_entries()
true_assignments = get_template_assignments(DATA_CONFIG['assignments_path'])
parser = Drain(tokenized_log_entries, 3, 100, 0.5)

parser.parse()
evaluator = Evaluator(true_assignments)
evaluator.print_all_discrepancies(parser.cluster_templates)
