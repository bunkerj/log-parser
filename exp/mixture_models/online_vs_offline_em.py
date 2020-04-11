"""
Log-likelihood comparison between online and offline EM.
"""
import numpy as np
from copy import deepcopy
from exp.mixture_models.utils import get_num_true_clusters
from global_utils import dump_results
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline

DATA_CONFIG = DataConfigs.OpenStack
N_INITIAL = 100
N_SAMPLE_SIZES = list(np.linspace(50, 2000, 40, dtype=np.int))

# Get relevant data
data_manager = DataManager(DATA_CONFIG)
log_entries = data_manager.get_tokenized_no_num_log_entries()
true_assignments = data_manager.get_true_assignments()
n_true_clusters = get_num_true_clusters(true_assignments)

offline_em_parser = MultinomialMixtureOnline(n_true_clusters, log_entries,
                                             False,
                                             epsilon=0.001)
online_em_parser = deepcopy(offline_em_parser)

offline_em_parser.perform_offline_em(log_entries, track_history=True)
offline_ll_history = offline_em_parser.get_log_likelihood_history()

online_em_parser.perform_online_batch_em(log_entries,
                                         fixed_iter=len(offline_ll_history) - 1,
                                         track_history=True)
online_ll_history = online_em_parser.get_log_likelihood_history()

dump_results('offline_vs_online_em_results.p', {
    'offline_log_likelihood': offline_ll_history,
    'online_log_likelihood': online_ll_history,
})
