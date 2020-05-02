"""
Evaluate and compare the timings and scores between EM, CEM, Online EM, and
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


def run_online_em(data_config, n_sample, training_sizes):
    # Get relevant data
    data_manager = DataManager(data_config)
    logs = data_manager.get_tokenized_no_num_logs()
    true_assignments = data_manager.get_true_assignments()
    n_true_clusters = get_num_true_clusters(true_assignments)

    ev = Evaluator(true_assignments)

    results = {
        'training_sizes': training_sizes,
        'scores': {'em': [], 'cem': [], 'online_em': [], 'online_cem': []},
        'timings': {'em': [], 'cem': [], 'online_em': [], 'online_cem': []},
    }

    for training_size in training_sizes:
        em_score = 0
        cem_score = 0
        online_em_score = 0
        online_cem_score = 0

        em_timing = 0
        cem_timing = 0
        online_em_timing = 0
        online_cem_timing = 0

        for _ in range(n_sample):
            print('Sample size: {}...'.format(training_size))

            # Randomly sample training logs
            training_indices = sample(range(len(logs)), k=training_size)
            training_logs = [logs[idx] for idx in
                             training_indices]

            # Fit parameters on all training logs
            em_parser = MultinomialMixtureOnline(logs,
                                                 n_true_clusters,
                                                 is_classification=False,
                                                 alpha=1.05,
                                                 beta=1.05)

            cem_parser = MultinomialMixtureOnline(logs,
                                                  n_true_clusters,
                                                  is_classification=True,
                                                  alpha=1.05,
                                                  beta=1.05)

            online_em_parser = MultinomialMixtureOnline(logs,
                                                        n_true_clusters,
                                                        is_classification=False,
                                                        alpha=1.05,
                                                        beta=1.05)

            online_cem_parser = MultinomialMixtureOnline(logs,
                                                         n_true_clusters,
                                                         is_classification=True,
                                                         alpha=1.05,
                                                         beta=1.05)

            cem_parser.set_parameters(em_parser.get_parameters())
            online_cem_parser.set_parameters(em_parser.get_parameters())
            online_em_parser.set_parameters(em_parser.get_parameters())

            # Run and get timings
            em_timing_tmp = time()
            em_parser.perform_offline_em(training_logs)
            em_timing += (time() - em_timing_tmp) / n_sample

            cem_timing_tmp = time()
            cem_parser.perform_offline_em(training_logs)
            cem_timing += (time() - cem_timing_tmp) / n_sample

            online_em_timing_tmp = time()
            online_em_parser.perform_online_batch_em(training_logs)
            online_em_timing += (time() - online_em_timing_tmp) / n_sample

            online_cem_timing_tmp = time()
            online_cem_parser.perform_online_batch_em(training_logs)
            online_cem_timing += (time() - online_cem_timing_tmp) / n_sample

            # Perform accuracy evaluations
            em_clusters = em_parser.get_clusters(logs)
            cem_clusters = cem_parser.get_clusters(logs)
            online_em_clusters = online_em_parser.get_clusters(logs)
            online_cem_clusters = online_cem_parser.get_clusters(logs)

            em_score += ev.get_impurity(em_clusters, []) / n_sample
            cem_score += ev.get_impurity(cem_clusters, []) / n_sample
            online_em_score += ev.get_impurity(online_em_clusters,
                                               []) / n_sample
            online_cem_score += ev.get_impurity(online_cem_clusters,
                                                []) / n_sample

        # Record scores
        results_acc = results['scores']
        results_acc['em'].append(em_score)
        results_acc['cem'].append(cem_score)
        results_acc['online_em'].append(online_em_score)
        results_acc['online_cem'].append(online_cem_score)

        # Record timings
        results_tim = results['timings']
        results_tim['em'].append(em_timing)
        results_tim['cem'].append(cem_timing)
        results_tim['online_em'].append(online_em_timing)
        results_tim['online_cem'].append(online_cem_timing)

    print('done!')
    return results


if __name__ == '__main__':
    n_sample = 3
    data_config = DataConfigs.HPC
    training_sizes = list(np.linspace(30, 2000, 5, dtype=np.int))

    results = run_online_em(data_config, n_sample, training_sizes)
    dump_results('online_em.p', results)
