"""
Evaluate how different initializations have an impact on impurity when using
the multinomial mixture model.
"""
from exp.mixture_models.utils import get_log_labels
from src.parsers.multinomial_mixture import MultinomialMixture
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from src.utils import get_template_assignments

DATA_CONFIG = DataConfigs.Proxifier

data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_log_entries()
true_assignments = get_template_assignments(DATA_CONFIG['assignments_path'])
num_true_clusters = len(set(log_data[-1] for log_data in true_assignments))
parser = MultinomialMixture(tokenized_log_entries, num_true_clusters)
evaluator = Evaluator(true_assignments)

log_labels = get_log_labels(true_assignments, 0)
parser.label_logs(log_labels)

parser.parse()
parser.print_cluster_samples(5)
impurity = evaluator.get_impurity(parser.cluster_templates,
                                  parser.labeled_indices)

print('Final Impurity: {}'.format(impurity))
