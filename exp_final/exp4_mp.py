"""
On using the 50k log dataset from BGL and 25k logs from Linux, we would perform
a run and compare the use of the coreset to choose logs that should be labeled
and compare performance against the heuristic where logs are uniformly sampled.
This would be done for different numbers of labels.
"""
import numpy as np
import multiprocessing as mp
from time import time
from exp_final.utils import get_log_sample, get_coreset
from global_utils import dump_results, get_log_labels, get_num_true_clusters
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from src.parsers.multinomial_mixture_vb import MultinomialMixtureVB


def run_exp4_full(data_config, label_counts, cs_proj_size,
                  subset_size, n_samples):
    results = {
        'cs_label_ami_samples': [],
        'cs_label_acc_samples': [],
        'rand_label_ami_samples': [],
        'rand_label_acc_samples': [],
        'cs_size': [],
    }

    data_manager = DataManager(data_config)
    logs, true_assignments = get_log_sample(data_manager, subset_size)
    n_clusters = get_num_true_clusters(true_assignments)
    ev = Evaluator(true_assignments)

    # Random labels
    with mp.Pool(mp.cpu_count()) as pool:
        for n_labels in label_counts:
            log_labels = get_log_labels(true_assignments, n_labels)
            args = (logs, n_clusters, log_labels, ev)
            arg_list = [args for _ in range(n_samples)]
            mp_results = pool.starmap(run_exp4_single, arg_list)
            ami_samples, acc_samples = list(zip(*mp_results))

            results['rand_label_ami_samples'].append(ami_samples)
            results['rand_label_acc_samples'].append(acc_samples)

    # Coreset labels
    with mp.Pool(mp.cpu_count()) as pool:
        for n_labels in label_counts:
            _, cs_logs, cs_indices \
                = get_coreset(logs, n_clusters, n_labels, cs_proj_size)
            cs_true_assignments = data_manager.get_reduced_assignments(
                cs_indices)
            log_labels = get_log_labels(cs_true_assignments, n_labels)
            results['cs_size'].append(len(cs_logs))

            args = (logs, n_clusters, log_labels, ev)
            arg_list = [args for _ in range(n_samples)]
            mp_results = pool.starmap(run_exp4_single, arg_list)
            ami_samples, acc_samples = list(zip(*mp_results))

            results['cs_label_ami_samples'].append(ami_samples)
            results['cs_label_acc_samples'].append(acc_samples)

    return results


def run_exp4_single(logs, n_clusters, log_labels, ev):
    np.random.seed()

    mm_random = MultinomialMixtureVB()
    mm_random.fit(logs, n_clusters, log_labels=log_labels)
    c_cs = mm_random.predict(logs)
    ami = ev.get_ami(c_cs)
    acc = ev.get_accuracy(c_cs)
    return ami, acc


if __name__ == '__main__':
    start_time = time()

    data_config = DataConfigs.BGL_FULL_FINAL
    subset_size = 50000
    n_samples = 1000
    label_counts = list(range(0, 250, 50))
    def_cs_proj_size = 1000

    filename = 'exp4_results.p'
    results = run_exp4_full(data_config, label_counts, def_cs_proj_size,
                            subset_size, n_samples)
    dump_results(filename, results)

    minutes_taken = (time() - start_time) / 60
    print('\nTime taken: {:.4f} minutes'.format(minutes_taken))
