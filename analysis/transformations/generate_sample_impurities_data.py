"""
Generate a dataset containing the sample impurities for each of the specified
log datasets.
"""
import os
import pandas as pd
from global_utils import load_results
from src.data_config import DataConfigs
from global_constants import RESULTS_DIR
from analysis.constants import NAME, PERCENTAGE_LABELED, LAB_IMPURITY, \
    UNLAB_IMPURITY
from global_constants import N_LOGS, LABEL_COUNTS, LABELED_IMPURITIES_SAMPLES, \
    UNLABELED_IMPURITIES_SAMPLES

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
    PERCENTAGE_LABELED: [],
    LAB_IMPURITY: [],
    UNLAB_IMPURITY: [],
}

for data_config in data_configs:
    name = data_config['name']
    results = load_results('feedback_evaluation_mp.p')
    dataset_results = results[name]

    n_logs = dataset_results[N_LOGS]
    label_counts = dataset_results[LABEL_COUNTS]
    lab_samples = dataset_results[LABELED_IMPURITIES_SAMPLES]
    unlab_samples = dataset_results[UNLABELED_IMPURITIES_SAMPLES]

    for sample_idx in range(len(lab_samples)):
        lab_sample_values = lab_samples[sample_idx]
        unlab_sample_values = unlab_samples[sample_idx]

        for label_idx in range(len(label_counts)):
            label_count = label_counts[label_idx]
            lab_sample_value = lab_sample_values[label_idx]
            unlab_sample_value = unlab_sample_values[label_idx]
            p_label = label_count / n_logs

            data[NAME].append(name)
            data[PERCENTAGE_LABELED].append(p_label)
            data[LAB_IMPURITY].append(lab_sample_value)
            data[UNLAB_IMPURITY].append(unlab_sample_value)

path = os.path.join(RESULTS_DIR, 'sample_impurities.csv')

df = pd.DataFrame(data)
df.to_csv(path, index=False)
