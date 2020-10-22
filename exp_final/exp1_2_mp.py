"""
This experiment will be run on 2k sample datasets. For each of these datasets,
we plot performance as function of the amount of feedback.
"""
import numpy as np
import multiprocessing as mp
from time import time
from src.helpers.oracle import Oracle
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from src.parsers.multinomial_mixture_vb import MultinomialMixtureVB
from global_utils import dump_results, get_num_true_clusters, sample_log_labels


def get_single_eval(logs, true_assignments, ev, oracle,
                    n_clusters, n_labels, n_consts):
    np.random.seed()

    log_labels = sample_log_labels(true_assignments, n_labels)

    mm = MultinomialMixtureVB()
    mm.fit(logs, n_clusters, log_labels=log_labels)
    c_no_consts = mm.predict(logs)
    score_no_consts = ev.get_ami(c_no_consts)

    if n_consts == 0:
        return score_no_consts

    W = oracle.get_corr_constraints_matrix(
        parsed_clusters=c_no_consts,
        n_constraint_samples=n_consts,
        tokenized_logs=logs,
        weight=1e7)
    mm.fit(logs, n_clusters,
           log_labels=log_labels,
           p_weights=W,
           sample_resp=False)
    c_feedback = mm.predict(logs)
    score_consts = ev.get_ami(c_feedback)

    return score_consts


def run_exp1_3_dataset(data_config, label_counts, const_counts, n_samples):
    data_manager = DataManager(data_config)
    logs = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()
    oracle = Oracle(true_assignments)
    n_clusters = get_num_true_clusters(true_assignments)
    ev = Evaluator(true_assignments)

    results = {
        'samples': {
            'base': [],
            'labels': [],
            'constraints': [],
        },
        'feedback_counts': {
            'labels': label_counts,
            'constraints': const_counts,
        }
    }

    # Base
    with mp.Pool(mp.cpu_count()) as pool:
        args = (logs, true_assignments, ev, oracle,
                n_clusters, 0, 0)
        arg_list = [args for _ in range(n_samples)]
        samples_base = pool.starmap(get_single_eval, arg_list)
        results['samples']['base'] = samples_base

    # For labels
    with mp.Pool(mp.cpu_count()) as pool:
        for n_labels in label_counts:
            args = (logs, true_assignments, ev, oracle,
                    n_clusters, n_labels, 0)
            arg_list = [args for _ in range(n_samples)]
            samples_labels = pool.starmap(get_single_eval, arg_list)
            results['samples']['labels'].append(samples_labels)

    # For constraints
    with mp.Pool(mp.cpu_count()) as pool:
        for n_consts in const_counts:
            args = (logs, true_assignments, ev, oracle,
                    n_clusters, 0, n_consts)
            arg_list = [args for _ in range(n_samples)]
            samples_constraints = pool.starmap(get_single_eval, arg_list)
            results['samples']['constraints'].append(samples_constraints)

    return results


def run_exp1_2_full(data_configs, label_counts, constraint_counts, n_samples):
    results = {}
    for data_config in data_configs:
        name = data_config['name']
        print(name)
        results[name] = run_exp1_3_dataset(data_config, label_counts,
                                           constraint_counts, n_samples)
    return results


if __name__ == '__main__':
    start_time = time()

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

    label_counts = list(range(0, 201, 40))
    constraint_counts = list(range(0, 201, 40))
    n_samples = 5

    filename = 'exp1_2_results.p'
    results = run_exp1_2_full(data_configs, label_counts,
                              constraint_counts, n_samples)
    dump_results(filename, results)

    minutes_taken = (time() - start_time) / 60
    print('\nTime taken: {:.4f} minutes'.format(minutes_taken))
