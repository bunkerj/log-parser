"""
Plot the accuracy over the similarity measure.
"""
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_template_assignments
from src.parsers.drain import Drain
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager

DATA_CONFIG = DataConfigs.Proxifier

accuracies = []
type1_error_ratios = []
type2_error_ratios = []

sim_thresholds = np.arange(0.01, 1.0, 0.01)
true_assignments = get_template_assignments(DATA_CONFIG['assignments_path'])

for sim_threshold in sim_thresholds:
    evaluator = Evaluator(true_assignments)
    data_manager = DataManager(DATA_CONFIG)
    tokenized_log_entries = data_manager.get_tokenized_log_entries()
    parser = Drain(tokenized_log_entries, 8, 75, sim_threshold)
    parser.parse()
    cluster_templates = parser.cluster_templates

    evaluator.evaluate(cluster_templates)
    type1_error_ratio = evaluator.get_type1_error_ratio()
    type2_error_ratio = evaluator.get_type2_error_ratio()
    accuracy = 1 - type1_error_ratio - type2_error_ratio

    type1_error_ratios.append(type1_error_ratio)
    type2_error_ratios.append(type2_error_ratio)
    accuracies.append(accuracy)

plt.plot(sim_thresholds, accuracies)
plt.plot(sim_thresholds, type1_error_ratios)
plt.plot(sim_thresholds, type2_error_ratios)
plt.legend(['Total Accuracy', 'Type 1 Error Ratio', 'Type 2 Error Ratio'])
plt.title('Error over Similarity Threshold')
plt.ylabel('Percentage')
plt.xlabel('Similarity Threshold')
plt.grid()
plt.show()
