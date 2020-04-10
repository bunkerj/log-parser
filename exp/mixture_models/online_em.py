"""
Evaluate and compare the timings and accuracies between CEM, Online EM, and
Online CEM.
"""
import numpy as np
from time import time
from copy import deepcopy
from random import sample
from exp.mixture_models.utils import get_num_true_clusters
from global_utils import dump_results
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline

DATA_CONFIG = DataConfigs.Apache
N_INITIAL = 30
N_SAMPLE_SIZES = list(np.linspace(50, 2000, 40, dtype=np.int))

# Get relevant data
data_manager = DataManager(DATA_CONFIG)
log_entries = data_manager.get_tokenized_no_num_log_entries()
true_assignments = data_manager.get_true_assignments()
n_true_clusters = get_num_true_clusters(true_assignments)

# Find proper initialization
initial_indices = sample(range(len(log_entries)), k=N_INITIAL)
initial_log_entries = [log_entries[idx] for idx in initial_indices]

cem_parser = MultinomialMixtureOnline(n_true_clusters, log_entries, True)
online_cem_parser = MultinomialMixtureOnline(n_true_clusters, log_entries, True)
online_em_parser = MultinomialMixtureOnline(n_true_clusters, log_entries, False)

cem_parser.find_best_initialization(initial_log_entries)
online_cem_parser.find_best_initialization(initial_log_entries)
online_em_parser.find_best_initialization(initial_log_entries)

results = {
    'sample_sizes': N_SAMPLE_SIZES,
    'accuracies': {'cem': [], 'online_cem': [], 'online_em': []},
    'timings': {'cem': [], 'online_cem': [], 'online_em': []},
}

for sample_size in N_SAMPLE_SIZES:
    print('Sample size: {}...'.format(sample_size))

    # Randomly sample data based on sample_size
    sample_indices = sample(range(len(log_entries)), k=sample_size)
    sample_log_entries = [log_entries[idx] for idx in sample_indices]

    # Fit parameters on all data points within the sample
    sample_cem_parser = deepcopy(cem_parser)
    sample_online_cem_parser = deepcopy(online_cem_parser)
    sample_online_em_parser = deepcopy(online_em_parser)

    cem_timing = time()
    sample_cem_parser.perform_offline_em(sample_log_entries)
    cem_timing = time() - cem_timing

    online_cem_timing = time()
    sample_online_cem_parser.perform_online_batch_em(sample_log_entries)
    online_cem_timing = time() - online_cem_timing

    online_em_timing = time()
    sample_online_em_parser.perform_online_batch_em(sample_log_entries)
    online_em_timing = time() - online_em_timing

    # Perform accuracy evaluations
    evaluator = Evaluator(true_assignments)

    cem_clusters = sample_cem_parser.get_clusters(log_entries)
    online_cem_clusters = sample_online_cem_parser.get_clusters(log_entries)
    online_em_clusters = sample_online_em_parser.get_clusters(log_entries)

    cem_accuracy = evaluator.evaluate(cem_clusters)
    online_cem_accuracy = evaluator.evaluate(online_cem_clusters)
    online_em_accuracy = evaluator.evaluate(online_em_clusters)

    # Record accuracies
    results_acc = results['accuracies']
    results_acc['cem'].append(cem_accuracy)
    results_acc['online_cem'].append(online_cem_accuracy)
    results_acc['online_em'].append(online_em_accuracy)

    # Record timings
    results_tim = results['timings']
    results_tim['cem'].append(cem_timing)
    results_tim['online_cem'].append(online_cem_timing)
    results_tim['online_em'].append(online_em_timing)

dump_results('online_em_results.p', results)

print('done!')
