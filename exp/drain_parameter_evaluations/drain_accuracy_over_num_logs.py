import matplotlib.pyplot as plt
from src.helpers.data_manager import DataManager
from src.parsers.drain import Drain
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.utils import get_template_assignments

JUMP_SIZE = 10
DATA_CONFIG = DataConfigs.BGL

data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_log_entries()
true_assignments = get_template_assignments(DATA_CONFIG['assignments_path'])
parser = Drain(tokenized_log_entries, 3, 100, 0.5)

indices = []
accuracies = []
for end_idx in range(1, len(tokenized_log_entries) + 1):
    parser.single_parse()
    if end_idx % JUMP_SIZE == 0 or end_idx == len(tokenized_log_entries):
        parser.discover_cluster_templates()
        evaluator = Evaluator(true_assignments[:end_idx], parser.cluster_templates)
        accuracies.append(evaluator.evaluate())
        indices.append(end_idx)

print('Final Drain Accuracy: {}'.format(accuracies[-1]))

plt.plot(indices, accuracies)
plt.title('Drain Accuracy Over # Logs')
plt.ylabel('Percentage Accuracy')
plt.xlabel('# Logs')
plt.grid()
plt.show()
