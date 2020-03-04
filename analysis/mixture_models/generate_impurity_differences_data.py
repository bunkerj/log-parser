import os
import pandas as pd
from constants import RESULTS_DIR
from global_utils import load_results
from src.data_config import DataConfigs

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
    'name': [],
    'provided_labels_count': [],
    'impurity_pct_diff': [],
}

for data_config in data_configs:
    name = data_config['name']
    results = load_results(
        'feedback_eval_{}_{}s.p'.format(name.lower(), N_SAMPLES))
    labeled_impurities = results[name]['avg_labeled_impurities']
    unlabeled_impurities = results[name]['avg_unlabeled_impurities']
    label_counts = results[name]['label_counts']

    for label_idx in range(len(label_counts)):
        data['name'].append(name)
        data['provided_labels_count'].append(label_counts[label_idx])
        data['impurity_pct_diff'].append(
            unlabeled_impurities[label_idx] - labeled_impurities[label_idx])

path = os.path.join(RESULTS_DIR, 'impurity_differences.csv')

df = pd.DataFrame(data)
df.to_csv(path, index=False)
