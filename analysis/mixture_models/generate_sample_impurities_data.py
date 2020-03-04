import os
import pandas as pd
from global_utils import load_results
from src.data_config import DataConfigs
from constants import RESULTS_DIR

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
    'dataset': [],
    'p_label': [],
    'lab_impurity': [],
    'unlab_impurity': [],
}

for data_config in data_configs:
    name = data_config['name']
    results = load_results('feedback_eval_{}_{}s.p'.format(name.lower(),
                                                           N_SAMPLES))
    n_logs = results[name]['n_logs']
    label_counts = results[name]['label_counts']
    lab_samples = results[name]['labeled_impurities_samples']
    unlab_samples = results[name]['unlabeled_impurities_samples']

    for sample_idx in range(len(lab_samples)):
        lab_sample_values = lab_samples[sample_idx]
        unlab_sample_values = unlab_samples[sample_idx]

        for label_idx in range(len(label_counts)):
            label_count = label_counts[label_idx]
            lab_sample_value = lab_sample_values[label_idx]
            unlab_sample_value = unlab_sample_values[label_idx]
            p_label = label_count / n_logs

            data['dataset'].append(name)
            data['p_label'].append(p_label)
            data['lab_impurity'].append(lab_sample_value)
            data['unlab_impurity'].append(unlab_sample_value)

path = os.path.join(RESULTS_DIR, 'sample_impurities.csv')

df = pd.DataFrame(data)
df.to_csv(path, index=False)
