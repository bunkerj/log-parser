"""
Print and save the accuracies of a single Drain run on a target dataset
(DATA_CONFIG).
"""
import matplotlib.pyplot as plt
from time import time
from src.parsers.enhanced_drain import EnhancedDrain
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from src.utils import get_template_assignments

DATA_CONFIG = DataConfigs.Proxifier

data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_log_entries()
true_assignments = get_template_assignments(DATA_CONFIG['assignments_path'])
parser = EnhancedDrain(tokenized_log_entries, 8.39, 34, 0.24, 0.18)

start_time = time()
parser.parse()
minutes_to_parse = (time() - start_time) / 60

start_time = time()
evaluator = Evaluator(true_assignments)
accuracy = evaluator.evaluate(parser.cluster_templates)
minutes_to_evaluate = (time() - start_time) / 60

print('Final Drain Accuracy: {}'.format(accuracy))
print('Time to parse: {:.5f} minutes'.format(minutes_to_parse))
print('Time to evaluate: {:.5f} minutes'.format(minutes_to_evaluate))
print('Total time taken: {:.5f} minutes'.format(
    minutes_to_parse + minutes_to_evaluate))

parser.plot_dendrogram(leaf_font_size=5,
                       leaf_rotation=67,
                       truncate_mode=None,
                       orientation='right')
plt.show()
