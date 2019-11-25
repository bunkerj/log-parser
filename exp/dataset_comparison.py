import os
import pickle
from src.methods.iplom import Iplom
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.parameter_grid_searcher import ParameterGridSearcher

RESULTS_DIR = '../results'

data_set_configs = [
    DataConfigs.Android,
    DataConfigs.Apache,
    DataConfigs.BGL,
    DataConfigs.Hadoop,
    DataConfigs.HDFS,
    DataConfigs.HealthApp,
    DataConfigs.HPC,
    DataConfigs.Linux,
    DataConfigs.Mac,
    DataConfigs.OpenSSH,
    DataConfigs.OpenStack,
    DataConfigs.Proxifier,
    DataConfigs.Spark,
    DataConfigs.Thunderbird,
    DataConfigs.Windows,
    DataConfigs.Zookeeper,
]

parameter_ranges_dict = {
    'file_threshold': (0, 0.15, 0.05),
    'partition_threshold': (0, 0.35, 0.05),
    'lower_bound': (0.1, 0.35, 0.05),
    'upper_bound': (0.9, 1, 1),
    'goodness_threshold': (0.3, 0.65, 0.05)
}

final_best_accuracies = {}

for data_set_config in data_set_configs:
    parameter_grid_searcher = ParameterGridSearcher(data_set_config, parameter_ranges_dict)
    parameter_grid_searcher.search()

    iplom = Iplom(data_set_config, **parameter_grid_searcher.best_parameters_dict)
    iplom.parse()

    evaluator = Evaluator(data_set_config, iplom.cluster_templates)
    iplom_accuracy = evaluator.evaluate()

    print('Final IPLoM {} Accuracy: {}'.format(data_set_config['name'], iplom_accuracy))
    final_best_accuracies[data_set_config['name']] = iplom_accuracy

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
pickle.dump(final_best_accuracies, open('../results/dataset_comparison.p', 'wb'))
