"""
Plot the accuracy over the maximum tree depth.
"""
import matplotlib.pyplot as plt
from src.parsers.drain import Drain
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager

DATA_CONFIG = DataConfigs.BGL

accuracies = []
tree_depths = list(range(3, 31, 1))
data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_log_entries()
true_assignments = data_manager.get_true_assignments()
evaluator = Evaluator(true_assignments)

for tree_depth in tree_depths:
    tokenized_log_entries = data_manager.get_tokenized_log_entries()
    parser = Drain(tokenized_log_entries, 3, tree_depth, 0.5)
    parser.parse()
    accuracies.append(evaluator.evaluate(parser.cluster_templates))

plt.plot(tree_depths, accuracies)
plt.title('Drain Accuracy Over Tree Depth')
plt.ylabel('Percentage Accuracy')
plt.xlabel('Tree Depth')
plt.grid()
plt.show()
