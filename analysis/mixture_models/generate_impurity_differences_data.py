"""
Generate a dataset containing the difference between the unsupervised impurity
means and the semi-supervised impurity means for each of the specified log
datasets.
"""
import os
import pandas as pd
from analysis.utils import get_avg_from_samples, get_var_from_samples
from global_constants import RESULTS_DIR
from global_utils import load_results
from src.data_config import DataConfigs
from analysis.constants import NAME, PROVIDED_LABELS_COUNT, \
    AVG_UNLAB_IMPURITY, AVG_LAB_IMPURITY, VAR_UNLAB_IMPURITY, VAR_LAB_IMPURITY
from global_constants import LABEL_COUNTS, LABELED_IMPURITIES_SAMPLES, \
    UNLABELED_IMPURITIES_SAMPLES

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
    AVG_UNLAB_IMPURITY: [],
    AVG_LAB_IMPURITY: [],
    VAR_UNLAB_IMPURITY: [],
    VAR_LAB_IMPURITY: [],
}

for data_config in data_configs:
    name = data_config['name']
    results = load_results(
        'feedback_eval_{}_{}s.p'.format(name.lower(), N_SAMPLES))
    labeled_impurity_samples = results[name][LABELED_IMPURITIES_SAMPLES]
    unlabeled_impurity_samples = results[name][UNLABELED_IMPURITIES_SAMPLES]
    label_count_values = results[name][LABEL_COUNTS]

    avg_lab_impurities = get_avg_from_samples(labeled_impurity_samples)
    avg_unlab_impurities = get_avg_from_samples(unlabeled_impurity_samples)
    var_lab_impurities = get_var_from_samples(labeled_impurity_samples)
    var_unlab_impurities = get_var_from_samples(unlabeled_impurity_samples)

    for label_idx in range(len(label_count_values)):
        data[NAME].append(name)
        data[PROVIDED_LABELS_COUNT].append(label_count_values[label_idx])
        data[AVG_UNLAB_IMPURITY].append(avg_unlab_impurities[label_idx])
        data[AVG_LAB_IMPURITY].append(avg_lab_impurities[label_idx])
        data[VAR_UNLAB_IMPURITY].append(var_unlab_impurities[label_idx])
        data[VAR_LAB_IMPURITY].append(var_lab_impurities[label_idx])

path = os.path.join(RESULTS_DIR, 'impurity_differences.csv')

df = pd.DataFrame(data)
df.to_csv(path, index=False)
