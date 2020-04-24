"""
Run an experiment to compare online EM vs online CEM where we do a single pass
over the data and provide true labels at regular intervals. An impurity measure
is then computed over the full dataset to compare both results.
"""
from time import time
from copy import deepcopy
from random import sample
from collections import defaultdict
from exp.mixture_models.utils import get_num_true_clusters, get_log_labels
from global_utils import dump_results, shuffle_same_order
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline

N_INIT = 500
N_RESTARTS = 10
N_SAMPLES = 20
N_LABELS = 1000
N_TEST = 2000
LABEL_INTERVAL = 1500
EVAL_INTERVAL = 3000
N_LOGS = 30000

DATA_CONFIG = DataConfigs.BGL_FULL

data_manager = DataManager(DATA_CONFIG)
log_entries = data_manager.get_tokenized_no_num_log_entries()
true_assignments = data_manager.get_true_assignments()
log_entries, true_assignments = shuffle_same_order(log_entries[:N_LOGS],
                                                   true_assignments[:N_LOGS])
n_true_clusters = get_num_true_clusters(true_assignments)

test_indices = sample(range(len(log_entries)), N_TEST)
test_logs = [log_entries[idx] for idx in test_indices]

ev = Evaluator(true_assignments)

results = {'lab_online_em': defaultdict(int),
           'unlab_online_em': defaultdict(int),
           'lab_online_cem': defaultdict(int)}

start_time = time()

for sample_idx in range(N_SAMPLES):
    lab_on_em_parser = MultinomialMixtureOnline(log_entries,
                                                n_true_clusters,
                                                is_classification=False,
                                                alpha=1.05,
                                                beta=1.05)

    lab_on_em_parser.find_best_initialization(log_entries, N_INIT, N_RESTARTS)

    unlab_on_em_parser = deepcopy(lab_on_em_parser)
    lab_on_cem_parser = deepcopy(lab_on_em_parser)
    lab_on_cem_parser.is_classification = True

    print('Start streaming...')
    start_iter = time()

    for log_idx_1, tokenized_log in enumerate(log_entries, start=1):
        lab_on_em_parser.perform_online_em(tokenized_log)
        unlab_on_em_parser.perform_online_em(tokenized_log)
        lab_on_cem_parser.perform_online_em(tokenized_log)

        if log_idx_1 == 1 or log_idx_1 % LABEL_INTERVAL == 0:
            label_indices = sample(range(len(log_entries)), N_LABELS)
            log_labels = get_log_labels(true_assignments, N_LABELS)
            lab_on_em_parser.label_logs(log_labels, log_entries)
            lab_on_cem_parser.label_logs(log_labels, log_entries)

        if log_idx_1 == 1 or log_idx_1 % EVAL_INTERVAL == 0:
            print('*** Eval @ iter {} ***'.format(log_idx_1))
            lab_on_em_clusters = lab_on_em_parser.get_clusters(test_logs)
            unlab_on_em_clusters = unlab_on_em_parser.get_clusters(test_logs)
            lab_on_cem_clusters = lab_on_cem_parser.get_clusters(test_logs)

            lab_on_em_imp = ev.get_impurity(lab_on_em_clusters, [])
            unlab_on_em_imp = ev.get_impurity(unlab_on_em_clusters, [])
            on_cem_imp = ev.get_impurity(lab_on_cem_clusters, [])

            results['lab_online_em'][log_idx_1] += lab_on_em_imp / N_SAMPLES
            results['unlab_online_em'][log_idx_1] += unlab_on_em_imp / N_SAMPLES
            results['lab_online_cem'][log_idx_1] += on_cem_imp / N_SAMPLES

        if log_idx_1 % 100 == 0:
            print('Iter {}: {:.4g} seconds'.format(
                log_idx_1, time() - start_iter))
            start_iter = time()

    print('End streaming.')

print('Time taken: {}'.format(time() - start_time))

result_filename = 'periodic_labeling_online_em_{}_{}s.p'.format(
    DATA_CONFIG['name'].lower(), N_SAMPLES)
dump_results(result_filename, results)
