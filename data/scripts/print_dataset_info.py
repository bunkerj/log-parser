from sklearn import metrics
from statistics import mean, stdev
from global_utils import get_num_true_clusters
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.utils import get_vocabulary_indices, get_token_counts_batch

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
]

for data_config in data_configs:
    name = data_config['name']
    data_manager = DataManager(data_config)
    logs = data_manager.get_tokenized_logs()
    logs_raw = data_manager.get_tokenized_raw_logs()
    n_logs = len(logs)

    true_assignments = data_manager.get_true_assignments()
    n_clusters = get_num_true_clusters(true_assignments)
    labels = [ta[-1] for ta in true_assignments]

    unique_log_set = set([' '.join(log) for log in logs])
    n_unique_logs = len(unique_log_set)

    v_indices = get_vocabulary_indices(logs)
    v_indices_raw = get_vocabulary_indices(logs_raw)
    n_vocab = len(v_indices)
    n_vocab_raw = len(v_indices_raw)

    C = get_token_counts_batch(logs, v_indices)
    C_raw = get_token_counts_batch(logs_raw, v_indices_raw)

    silhouette = metrics.silhouette_score(C, labels)
    silhouette_raw = metrics.silhouette_score(C_raw, labels)
    db = metrics.davies_bouldin_score(C, labels)
    db_raw = metrics.davies_bouldin_score(C_raw, labels)

    log_lengths = [len(log) for log in logs]
    log_raw_lengths = [len(log) for log in logs_raw]

    avg_tokens_per_log_cont = mean(log_lengths)
    avg_tokens_per_log_raw = mean(log_raw_lengths)
    std_tokens_per_log_cont = stdev(log_lengths)
    std_tokens_per_log_raw = stdev(log_raw_lengths)
    raw_vs_cont_ratio = avg_tokens_per_log_raw / avg_tokens_per_log_cont

    print('------ {} ------'.format(name))
    print('n_clusters: {}'.format(n_clusters))
    print('n_unique_logs: {}'.format(n_unique_logs))
    print('n_vocab_raw: {}'.format(n_vocab_raw))
    print('n_vocab: {}'.format(n_vocab))
    print('n_logs: {}'.format(n_logs))
    print('avg_tokens_per_log_raw: {}'.format(avg_tokens_per_log_raw))
    print('avg_tokens_per_log_cont: {}'.format(avg_tokens_per_log_cont))
    print('stdev_tok_per_log_raw: {:.4f}'.format(std_tokens_per_log_raw))
    print('stdev_tok_per_log_cont: {:.4f}'.format(std_tokens_per_log_cont))
    print('raw_vs_cont_ratio: {:.4f}'.format(raw_vs_cont_ratio))
    print('silhouette: {:.4f}'.format(silhouette))
    print('silhouette_raw: {:.4f}'.format(silhouette_raw))
    print('db: {:.4f}'.format(db))
    print('db_raw: {:.4f}'.format(db_raw))
    print()
