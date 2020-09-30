"""
This experiment would be performed on the 250k log BGL dataset. Drain is used as
a competing baseline. Using a suitable coreset upper bound size and coreset
projection size from experiment 2, we would record NMI as a function of: labels
and pairwise constraints (using pairwise constraints as a corrective measure as
described in experiment 1).
"""
import numpy as np
import multiprocessing as mp
from time import time
from exp_final.utils import get_coreset
from global_utils import dump_results, sample_log_labels, get_num_true_clusters
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from src.helpers.oracle import Oracle
from src.parsers.drain import Drain
from src.parsers.multinomial_mixture_vb import MultinomialMixtureVB


def run_exp3_full(data_config, label_counts, constraint_counts,
                  cs_ub_size, cs_proj_size, n_samples):
    results = {
        'label_counts': label_counts,
        'label_ami_samples': [],
        'label_acc_samples': [],
        'constraint_counts': constraint_counts,
        'constraint_ami_samples': [],
        'constraint_acc_samples': [],
        'drain_ami': None,
        'drain_acc': None,
    }

    data_manager = DataManager(data_config)
    logs = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()
    n_clusters = get_num_true_clusters(true_assignments)
    ev = Evaluator(true_assignments)

    cs_w, cs_logs, cs_indices \
        = get_coreset(logs, n_clusters, cs_ub_size, cs_proj_size)
    cs_true_assignments = data_manager.get_reduced_assignments(cs_indices)
    oracle = Oracle(cs_true_assignments)
    results['cs_size'] = len(cs_logs)

    # Drain performance
    parser = Drain(logs, 4, 100, 0.4)
    parser.parse()
    c_drain = parser.cluster_templates
    results['drain_ami'] = ev.get_ami(c_drain)
    results['drain_acc'] = ev.get_accuracy(c_drain)

    # MM_VB labels only
    with mp.Pool(mp.cpu_count()) as pool:
        for n_labels in label_counts:
            args = (logs, cs_logs, cs_w, cs_true_assignments, n_clusters,
                    oracle, ev, n_labels, 0)
            arg_list = [args for _ in range(n_samples)]
            mp_results = pool.starmap(run_exp3_single, arg_list)
            ami_samples, acc_samples = list(zip(*mp_results))
            results['label_ami_samples'].append(ami_samples)
            results['label_acc_samples'].append(acc_samples)

    # MM_VB constraints only
    with mp.Pool(mp.cpu_count()) as pool:
        for n_constraints in constraint_counts:
            args = (logs, cs_logs, cs_w, cs_true_assignments, n_clusters,
                    oracle, ev, 300, n_constraints)
            arg_list = [args for _ in range(n_samples)]
            mp_results = pool.starmap(run_exp3_single, arg_list)
            ami_samples, acc_samples = list(zip(*mp_results))
            results['constraint_ami_samples'].append(ami_samples)
            results['constraint_acc_samples'].append(acc_samples)

    return results


def run_exp3_single(logs, cs_logs, cs_w, cs_true_assignments, n_clusters,
                    oracle, ev, n_labels, n_constraints):
    np.random.seed()

    log_labels = sample_log_labels(cs_true_assignments, n_labels)

    # Original fit
    mm = MultinomialMixtureVB()
    mm.fit(cs_logs, n_clusters,
           cs_weights=cs_w,
           log_labels=log_labels)
    c_cs = mm.predict(cs_logs)

    # Corrected fit
    W = oracle.get_corr_constraints_matrix(
        parsed_clusters=c_cs,
        n_constraint_samples=n_constraints,
        tokenized_logs=cs_logs,
        weight=1e7)
    mm.fit(cs_logs, n_clusters,
           cs_weights=cs_w,
           log_labels=log_labels,
           p_weights=W,
           sample_resp=False)
    c_cs = mm.predict(logs)
    ami = ev.get_ami(c_cs)
    acc = ev.get_accuracy(c_cs)

    return ami, acc


if __name__ == '__main__':
    start_time = time()

    data_config = DataConfigs.BGL_FULL_FINAL
    n_samples = 1000
    label_counts = list(range(0, 2001, 400))
    constraint_counts = list(range(0, 2001, 400))
    def_cs_ub_size = 2000
    def_cs_proj_size = 1000

    filename = 'exp3_results_test.p'
    results = run_exp3_full(data_config, label_counts, constraint_counts,
                            def_cs_ub_size, def_cs_proj_size, n_samples)
    dump_results(filename, results)

    minutes_taken = (time() - start_time) / 60
    print('\nTime taken: {:.4f} minutes'.format(minutes_taken))
