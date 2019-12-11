"""
Plot the Drain accuracy for a specific dataset (DATA_CONFIG) evaluated at every
increment (JUMP_SIZE). This procedure is repeated several times (N_RUNS) and
each run is performed on a shuffled version of the dataset.
"""
import random
import matplotlib.pyplot as plt
from src.helpers.data_manager import DataManager
from src.parsers.drain import Drain
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.utils import get_template_assignments

JUMP_SIZE = 10
N_RUNS = 20
DATA_CONFIG = DataConfigs.BGL

data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_log_entries()
true_assignments = get_template_assignments(DATA_CONFIG['assignments_path'])

accuracies = []
end_indices = []
for run in range(N_RUNS):
    accuracy_idx = 0
    rand_float = random.random()
    random.shuffle(tokenized_log_entries, lambda: rand_float)
    random.shuffle(true_assignments, lambda: rand_float)
    parser = Drain(tokenized_log_entries, 3, 100, 0.5)
    for end_idx in range(1, len(tokenized_log_entries) + 1):
        parser.single_parse()
        if end_idx % JUMP_SIZE == 0 or end_idx == len(tokenized_log_entries):
            parser.discover_cluster_templates()
            evaluator = Evaluator(true_assignments[:end_idx],
                                  parser.cluster_templates)
            accuracy = evaluator.evaluate()
            if run == 0:
                accuracies.append(accuracy / N_RUNS)
                end_indices.append(end_idx)
            else:
                accuracies[accuracy_idx] += accuracy / N_RUNS
            accuracy_idx += 1

print('Final Drain Accuracy: {}'.format(accuracies[-1]))

plt.plot(end_indices, accuracies)
plt.title('Drain Accuracy Over # Logs')
plt.ylabel('Percentage Accuracy')
plt.xlabel('# Logs')
plt.grid()
plt.show()
