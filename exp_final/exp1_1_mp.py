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
from exp.utils import get_extended_cs
from exp_final.utils import get_coreset
from src.helpers.oracle import Oracle
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from src.parsers.multinomial_mixture_vb import MultinomialMixtureVB
from global_utils import dump_results, sample_log_labels, \
    get_num_true_clusters, get_labeled_indices, get_reduced_assignments


def get_clustering_evaluations(logs, cs_true_assignments, ev, oracle,
                               n_clusters, n_labels, n_consts, cs_w, cs_logs):
    np.random.seed()

    log_labels = sample_log_labels(cs_true_assignments, n_labels)
    labeled_indices = get_labeled_indices(log_labels)

    # Baseline
    mm = MultinomialMixtureVB()
    mm.fit(cs_logs, n_clusters, cs_weights=cs_w)
    c_base = mm.predict(logs)

    # Baseline + constraints
    W_const = oracle.get_corr_constraints_matrix(
        parsed_clusters=mm.predict(cs_logs),
        n_constraint_samples=n_consts,
        tokenized_logs=cs_logs,
        weight=1e7)
    mm.fit(cs_logs, n_clusters,
           cs_weights=cs_w,
           p_weights=W_const,
           max_iter=25,
           sample_resp=False)
    c_const = mm.predict(logs)

    # Baseline + labels
    mm_lab = MultinomialMixtureVB()
    mm_lab.fit(cs_logs, n_clusters, log_labels=log_labels, cs_weights=cs_w)
    c_lab = mm_lab.predict(logs)

    # Baseline + labels + constraints
    W_lab_const = oracle.get_corr_constraints_matrix(
        parsed_clusters=mm_lab.predict(cs_logs),
        n_constraint_samples=n_consts,
        tokenized_logs=cs_logs,
        weight=1e7)
    mm_lab.fit(cs_logs, n_clusters,
               log_labels=log_labels,
               p_weights=W_lab_const,
               max_iter=25,
               sample_resp=False,
               cs_weights=cs_w)
    c_lab_const = mm_lab.predict(logs)

    score_base = ev.get_ami(c_base, labeled_indices)
    score_const = ev.get_ami(c_const, labeled_indices)
    score_lab = ev.get_ami(c_lab, labeled_indices)
    score_lab_const = ev.get_ami(c_lab_const, labeled_indices)

    return score_base, score_const, score_lab, score_lab_const


def run_exp1_single_dataset_mp(data_config, n_labels, n_consts, n_samples):
    data_manager = DataManager(data_config)
    logs = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()
    n_clusters = get_num_true_clusters(true_assignments)
    ev = Evaluator(true_assignments)

    cs_w, cs_logs, cs_indices \
        = get_coreset(logs, n_clusters, 200, 2000)

    cs_w, cs_logs, cs_indices = \
        get_extended_cs(cs_w, cs_logs, cs_indices)

    cs_true_assignments = get_reduced_assignments(cs_indices, true_assignments)
    oracle = Oracle(cs_true_assignments)

    with mp.Pool(mp.cpu_count()) as pool:
        args = (logs, cs_true_assignments, ev, oracle,
                n_clusters, n_labels, n_consts, cs_w, cs_logs)
        arg_list = [args for _ in range(n_samples)]
        mp_results = pool.starmap(get_clustering_evaluations, arg_list)

    clustering_results = list(zip(*mp_results))

    return {
        'score_base_samples': clustering_results[0],
        'score_const_samples': clustering_results[1],
        'score_lab_samples': clustering_results[2],
        'score_lab_const_samples': clustering_results[3],
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
        DataConfigs.Mac,
        DataConfigs.OpenSSH,
        DataConfigs.OpenStack,
        DataConfigs.Proxifier,
        DataConfigs.Spark,
        DataConfigs.Thunderbird,
        DataConfigs.Windows,
        DataConfigs.Zookeeper,
    ]

    n_labels = 200
    n_consts = 200
    n_samples = 1000

    filename = 'exp1_results.p'
    results = run_exp1_full(data_configs, n_labels, n_consts, n_samples)
    dump_results(filename, results)

    minutes_taken = (time() - start_time) / 60
    print('\nTime taken: {:.4f} minutes'.format(minutes_taken))
