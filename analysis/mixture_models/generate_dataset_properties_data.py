"""
Generate a dataset containing features for each of the specified log datasets.
"""
import os
import pandas as pd
from global_constants import RESULTS_DIR
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from exp.mixture_models.utils import get_avg_gini_impurity
from src.utils import get_vocabulary_indices, get_token_counts
from exp.mixture_models.utils import get_num_true_clusters, normalize_matrix
from analysis.utils import get_intra_cluster_spread, get_inter_cluster_spread, \
    split_counts_per_cluster
from analysis.constants import NAME, VOCAB_SIZE, TRUE_CLUSTER_COUNT, \
    AVG_FREQ_GINI, INTRA_CLUSTER_SPREAD, INTER_CLUSTER_SPREAD

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
    VOCAB_SIZE: [],
    TRUE_CLUSTER_COUNT: [],
    AVG_FREQ_GINI: [],
    INTRA_CLUSTER_SPREAD: [],
    INTER_CLUSTER_SPREAD: [],
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

    data[NAME].append(name)
    data[VOCAB_SIZE].append(len(v_indices))
    data[TRUE_CLUSTER_COUNT].append(get_num_true_clusters(true_assignments))
    data[AVG_FREQ_GINI].append(get_avg_gini_impurity(C_probabilities, 1))
    data[INTRA_CLUSTER_SPREAD].append(
        get_intra_cluster_spread(count_cluster_split, C))
    data[INTER_CLUSTER_SPREAD].append(
        get_inter_cluster_spread(count_cluster_split, C))

path = os.path.join(RESULTS_DIR, 'dataset_properties.csv')

df = pd.DataFrame(data)
df.to_csv(path, index=False)
