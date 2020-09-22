"""
This experiment will be run on the following eight 2k sample datasets. For each
of these datasets, a bar chart will display a score between the following
algorithms: baseline, baseline + 5% labels (100 labels), and baseline + 5%
labels + 5% pairwise constraints (up to 100 pairwise-constraints). To get the
pairwise constraints, the baseline + 5% labels algorithm will be used to provide
clusters and the pairwise constraints will be randomly sampled to correct the
initial clustering.
"""
import numpy as np
import multiprocessing as mp
from time import time
from global_utils import dump_results, get_log_labels, get_num_true_clusters
from src.helpers.oracle import Oracle
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.parsers.multinomial_mixture_vb import MultinomialMixtureVB


def get_clustering_evaluations(logs, true_assignments, oracle,
                               n_clusters, n_labels, n_consts):
    np.random.seed()

    log_labels = get_log_labels(true_assignments, n_labels)

    # Baseline
    mm = MultinomialMixtureVB()
    mm.fit(logs, n_clusters)
    c_base = mm.predict(logs)

    # Baseline + 5% labels (100 labels)
    mm_lab = MultinomialMixtureVB()
    mm_lab.fit(logs, n_clusters, log_labels=log_labels)
    c_lab = mm_lab.predict(logs)

    # Baseline + labels + constraints
    mm_lab_const = MultinomialMixtureVB()
    W = oracle.get_corr_constraints_matrix(
        parsed_clusters=c_lab,
        n_constraint_samples=n_consts,
        tokenized_logs=logs,
        weight=10)
    mm_lab_const.fit(logs, n_clusters, log_labels=log_labels, p_weights=W)
    c_lab_const = mm_lab_const.predict(logs)

    return c_base, c_lab, c_lab_const, log_labels


def run_exp1_single_dataset_mp(data_config, n_labels, n_consts, n_samples):
    data_manager = DataManager(data_config)
    logs = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()
    oracle = Oracle(true_assignments)
    n_clusters = get_num_true_clusters(true_assignments)

    with mp.Pool(mp.cpu_count()) as pool:
        args = (logs, true_assignments, oracle,
                n_clusters, n_labels, n_consts)
        arg_list = [args for _ in range(n_samples)]
        mp_results = pool.starmap(get_clustering_evaluations, arg_list)

    clustering_results = list(zip(*mp_results))

    return {
        'clustering_base_samples': clustering_results[0],
        'clustering_lab_samples': clustering_results[1],
        'clustering_lab_const_samples': clustering_results[2],
        'log_labels_samples': clustering_results[3],
    }


def run_exp1_full(data_configs, n_labels, n_consts, n_samples):
    results = {}
    for data_config in data_configs:
        name = data_config['name']
        print(name)
        results[name] = run_exp1_single_dataset_mp(data_config, n_labels,
                                                   n_consts, n_samples)
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
    ]

    n_labels = 0
    n_consts = 100
    n_samples = 1000

    filename = 'exp1_results.p'
    results = run_exp1_full(data_configs, n_labels, n_consts, n_samples)
    dump_results(filename, results)

    minutes_taken = (time() - start_time) / 60
    print('\nTime taken: {:.4f} minutes'.format(minutes_taken))
