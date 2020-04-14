"""
Log-likelihood comparison between online and offline EM.
"""
from random import sample
from copy import deepcopy
from exp.mixture_models.utils import get_num_true_clusters
from global_utils import dump_results
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline

N_INIT = 100

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

results = {}

for data_config in data_configs:
    name = data_config['name']
    results[name] = {'offline': None, 'online': None}

    print('Running for {}...'.format(name))

    data_manager = DataManager(data_config)
    log_entries = data_manager.get_tokenized_no_num_log_entries()
    true_assignments = data_manager.get_true_assignments()
    n_true_clusters = get_num_true_clusters(true_assignments)

    init_indices = sample(range(len(log_entries)), k=N_INIT)
    init_log_entries = [log_entries[idx] for idx in init_indices]

    online_em_parser = MultinomialMixtureOnline(log_entries,
                                                n_true_clusters,
                                                False,
                                                epsilon=0.01)
    online_em_parser.find_best_initialization(init_log_entries)

    offline_em_parser = deepcopy(online_em_parser)
    offline_em_parser.init_sufficient_stats()

    offline_em_parser.perform_offline_em(log_entries, track_history=True)
    offline_ll_history = offline_em_parser.get_log_likelihood_history()
    results[name]['offline'] = offline_ll_history

    online_em_parser.perform_online_batch_em(log_entries,
                                             len(offline_ll_history) - 1,
                                             track_history=True)
    online_ll_history = online_em_parser.get_log_likelihood_history()
    results[name]['online'] = online_ll_history

dump_results('offline_vs_online_em_results.p', results)
