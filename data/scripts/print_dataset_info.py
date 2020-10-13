from global_utils import get_num_true_clusters
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.utils import get_vocabulary_indices

data_configs = [
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
    DataConfigs.BGL_FULL_FINAL,
]

for data_config in data_configs:
    name = data_config['name']
    data_manager = DataManager(data_config)
    logs = data_manager.get_tokenized_logs()
    n_logs = len(logs)

    true_assignments = data_manager.get_true_assignments()
    n_clusters = get_num_true_clusters(true_assignments)

    event_set = set([' '.join(log) for log in logs])
    n_unique_logs = len(event_set)

    v_indices = get_vocabulary_indices(logs)
    n_vocab = len(v_indices)

    print(name)
    print('Number of clusters: {}'.format(n_clusters))
    print('Number of unique logs: {}'.format(n_unique_logs))
    print('Number of unique tokens: {}'.format(n_vocab))
    print('Number of logs: {}\n'.format(n_logs))
