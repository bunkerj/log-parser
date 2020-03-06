"""
Generate a dataset containing the difference between the unsupervised impurity
means and the semi-supervised impurity means for each of the specified log
datasets.
"""
import os
import pandas as pd
from global_constants import RESULTS_DIR
from global_utils import load_results
from src.data_config import DataConfigs
from analysis.constants import NAME, PROVIDED_LABELS_COUNT, IMPURITY_MEAN_DIFF
from global_constants import AVG_LABELED_IMPURITIES, AVG_UNLABELED_IMPURITIES, \
    LABEL_COUNTS

N_SAMPLES = 50

data_configs = [
    DataConfigs.Android,
    DataConfigs.Apache,
    DataConfigs.BGL,
    DataConfigs.Hadoop,
    DataConfigs.HDFS,
    DataConfigs.HealthApp,
    DataConfigs.HPC,
    DataConfigs.Linux,
    # DataConfigs.Mac,
    # DataConfigs.OpenSSH,
    # DataConfigs.OpenStack,
    # DataConfigs.Proxifier,
    # DataConfigs.Spark,
    # DataConfigs.Thunderbird,
    # DataConfigs.Windows,
    # DataConfigs.Zookeeper,
]

data = {
    NAME: [],
    PROVIDED_LABELS_COUNT: [],
    IMPURITY_MEAN_DIFF: [],
}

for data_config in data_configs:
    name = data_config['name']
    results = load_results(
        'feedback_eval_{}_{}s.p'.format(name.lower(), N_SAMPLES))
    labeled_impurities = results[name][AVG_LABELED_IMPURITIES]
    unlabeled_impurities = results[name][AVG_UNLABELED_IMPURITIES]
    label_counts = results[name][LABEL_COUNTS]

    for label_idx in range(len(label_counts)):
        data[NAME].append(name)
        data[PROVIDED_LABELS_COUNT].append(label_counts[label_idx])
        data[IMPURITY_MEAN_DIFF].append(
            unlabeled_impurities[label_idx] - labeled_impurities[label_idx])

path = os.path.join(RESULTS_DIR, 'impurity_differences.csv')

df = pd.DataFrame(data)
df.to_csv(path, index=False)
