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


def run_periodic_labeling_online_em(data_config, n_init, n_restarts, n_samples,
                                    n_labels, n_test, label_interval,
                                    eval_interval, n_logs):
    data_manager = DataManager(data_config)
    logs = data_manager.get_tokenized_no_num_logs()
    true_assignments = data_manager.get_true_assignments()
    logs, true_assignments = shuffle_same_order(logs[:n_logs],
                                                true_assignments[
                                                :n_logs])
    n_true_clusters = get_num_true_clusters(true_assignments)

    test_indices = sample(range(len(logs)), n_test)
    test_logs = [logs[idx] for idx in test_indices]

    ev = Evaluator(true_assignments)

    res = {'lab_online_em': defaultdict(int),
           'unlab_online_em': defaultdict(int),
           'lab_online_cem': defaultdict(int)}

    start_time = time()

    for sample_idx in range(n_samples):
        lab_on_em_parser = MultinomialMixtureOnline(logs,
                                                    n_true_clusters,
                                                    is_classification=False,
                                                    alpha=1.05,
                                                    beta=1.05)

        lab_on_em_parser.find_best_initialization(logs, n_init,
                                                  n_restarts)

        unlab_on_em_parser = deepcopy(lab_on_em_parser)
        lab_on_cem_parser = deepcopy(lab_on_em_parser)
        lab_on_cem_parser.is_classification = True

        print('Start streaming...')
        start_iter = time()

        for log_idx_1, tokenized_log in enumerate(logs, start=1):
            lab_on_em_parser.perform_online_em(tokenized_log)
            unlab_on_em_parser.perform_online_em(tokenized_log)
            lab_on_cem_parser.perform_online_em(tokenized_log)

            if log_idx_1 == 1 or log_idx_1 % label_interval == 0:
                label_indices = sample(range(len(logs)), n_labels)
                log_labels = get_log_labels(true_assignments, n_labels)
                lab_on_em_parser.label_logs(log_labels, logs)
                lab_on_cem_parser.label_logs(log_labels, logs)

            if log_idx_1 == 1 or log_idx_1 % eval_interval == 0:
                print('*** Eval @ iter {} ***'.format(log_idx_1))
                lab_on_em_clusters = lab_on_em_parser.get_clusters(test_logs)
                unlab_on_em_clusters = unlab_on_em_parser.get_clusters(
                    test_logs)
                lab_on_cem_clusters = lab_on_cem_parser.get_clusters(test_logs)

                lab_on_em_imp = ev.get_impurity(lab_on_em_clusters, [])
                unlab_on_em_imp = ev.get_impurity(unlab_on_em_clusters, [])
                on_cem_imp = ev.get_impurity(lab_on_cem_clusters, [])

                res['lab_online_em'][log_idx_1] += lab_on_em_imp / n_samples
                res['unlab_online_em'][log_idx_1] += unlab_on_em_imp / n_samples
                res['lab_online_cem'][log_idx_1] += on_cem_imp / n_samples

            if log_idx_1 % 100 == 0:
                print('Iter {}: {:.4g} seconds'.format(
                    log_idx_1, time() - start_iter))
                start_iter = time()

        print('End streaming.')

    print('Time taken: {}'.format(time() - start_time))
    return res


if __name__ == '__main__':
    n_init = 50
    n_restarts = 10
    n_samples = 3
    n_labels = 1000
    n_test = 2000
    label_interval = 2000
    eval_interval = 2500
    n_logs = 10000
    data_config = DataConfigs.BGL_FULL

    results = run_periodic_labeling_online_em(data_config, n_init, n_restarts,
                                              n_samples, n_labels, n_test,
                                              label_interval, eval_interval,
                                              n_logs)
    dump_results('periodic_labeling_online_em.p', results)
