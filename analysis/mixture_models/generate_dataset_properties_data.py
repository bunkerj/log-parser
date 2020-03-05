"""
The goal of this script is to create a dataset that includes key properties of
the various datasets that are being used for clustering.

The dataset properties are:
1) Vocabulary Size
2) Average Frequency Gini Impurity
3) True Cluster Count
"""
import os
import pandas as pd
from constants import RESULTS_DIR
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from exp.mixture_models.utils import get_avg_gini_impurity
from src.utils import get_vocabulary_indices, get_token_counts
from exp.mixture_models.utils import get_num_true_clusters, normalize_matrix
from analysis.utils import get_avg_intra_score, \
    get_avg_inter_score, \
    split_counts_per_cluster

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
    'vocab_size': [],
    'true_cluster_count': [],
    'avg_freq_gini': [],
    'intra_cluster_score': [],
    'inter_cluster_score': [],
}

for data_config in data_configs:
    name = data_config['name']
    data_manager = DataManager(data_config)
    tokenized_log_entries = data_manager.get_tokenized_no_num_log_entries()
    true_assignments = data_manager.get_true_assignments()

    v_indices = get_vocabulary_indices(tokenized_log_entries)
    C = get_token_counts(tokenized_log_entries, v_indices)
    C_probabilities = normalize_matrix(C, 1)
    count_cluster_split = split_counts_per_cluster(C, true_assignments)

    print('{}: {}'.format(name, len(v_indices)))

    data['name'].append(name)
    data['vocab_size'].append(len(v_indices))
    data['true_cluster_count'].append(get_num_true_clusters(true_assignments))
    data['avg_freq_gini'].append(get_avg_gini_impurity(C_probabilities, 1))
    data['intra_cluster_score'].append(get_avg_intra_score(count_cluster_split))
    data['inter_cluster_score'].append(get_avg_inter_score(count_cluster_split))

path = os.path.join(RESULTS_DIR, 'dataset_properties.csv')

df = pd.DataFrame(data)
df.to_csv(path, index=False)
