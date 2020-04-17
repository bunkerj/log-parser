"""
Evaluate and compare the timings and scores between CEM, Online EM, and
Online CEM.
"""
import numpy as np
from time import time
from random import sample
from exp.mixture_models.utils import get_num_true_clusters
from global_utils import dump_results
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline

N_SAMPLE = 20
DATA_CONFIG = DataConfigs.Apache
TRAINING_SIZES = list(np.linspace(50, 2000, 40, dtype=np.int))

# Get relevant data
data_manager = DataManager(DATA_CONFIG)
log_entries = data_manager.get_tokenized_no_num_log_entries()
true_assignments = data_manager.get_true_assignments()
n_true_clusters = get_num_true_clusters(true_assignments)

results = {
    'training_sizes': TRAINING_SIZES,
    'scores': {'cem': [], 'online_cem': [], 'online_em': []},
    'timings': {'cem': [], 'online_cem': [], 'online_em': []},
}

for training_size in TRAINING_SIZES:
    cem_score = 0
    online_cem_score = 0
    online_em_score = 0

    cem_timing = 0
    online_cem_timing = 0
    online_em_timing = 0

    for _ in range(N_SAMPLE):
        print('Sample size: {}...'.format(training_size))

        # Randomly sample training logs
        training_indices = sample(range(len(log_entries)), k=training_size)
        training_log_entries = [log_entries[idx] for idx in training_indices]

        # Fit parameters on all training logs
        cem_parser = MultinomialMixtureOnline(log_entries,
                                              n_true_clusters,
                                              is_classification=False,
                                              alpha=1.05,
                                              beta=1.05)

        online_cem_parser = MultinomialMixtureOnline(log_entries,
                                                     n_true_clusters,
                                                     is_classification=True,
                                                     alpha=1.05,
                                                     beta=1.05)

        online_em_parser = MultinomialMixtureOnline(log_entries,
                                                    n_true_clusters,
                                                    is_classification=False,
                                                    alpha=1.05,
                                                    beta=1.05)

        online_cem_parser.set_parameters(cem_parser.get_parameters())
        online_em_parser.set_parameters(cem_parser.get_parameters())

        cem_timing_tmp = time()
        cem_parser.perform_offline_em(training_log_entries)
        cem_timing += (time() - cem_timing_tmp) / N_SAMPLE

        online_cem_timing_tmp = time()
        online_cem_parser.perform_online_batch_em(training_log_entries)
        online_cem_timing += (time() - online_cem_timing_tmp) / N_SAMPLE

        online_em_timing_tmp = time()
        online_em_parser.perform_online_batch_em(training_log_entries)
        online_em_timing += (time() - online_em_timing_tmp) / N_SAMPLE

        # Perform accuracy evaluations
        evaluator = Evaluator(true_assignments)

        cem_clusters = cem_parser.get_clusters(log_entries)
        online_cem_clusters = online_cem_parser.get_clusters(log_entries)
        online_em_clusters = online_em_parser.get_clusters(log_entries)

        cem_score += evaluator.get_impurity(cem_clusters,
                                            []) / N_SAMPLE
        online_cem_score += evaluator.get_impurity(online_cem_clusters,
                                                   []) / N_SAMPLE
        online_em_score += evaluator.get_impurity(online_em_clusters,
                                                  []) / N_SAMPLE

    # Record scores
    results_acc = results['scores']
    results_acc['cem'].append(cem_score)
    results_acc['online_cem'].append(online_cem_score)
    results_acc['online_em'].append(online_em_score)

    # Record timings
    results_tim = results['timings']
    results_tim['cem'].append(cem_timing)
    results_tim['online_cem'].append(online_cem_timing)
    results_tim['online_em'].append(online_em_timing)

dump_results('online_em_results.p', results)

print('done!')
