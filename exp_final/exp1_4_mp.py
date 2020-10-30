"""
This experiment will be run on 2k sample datasets. For each of these datasets,
we plot performance for both Drain and MultinomialVB.
"""
import numpy as np
import multiprocessing as mp
from time import time
from exp_final.utils import permute_logs
from src.parsers.drain import Drain
from src.helpers.oracle import Oracle
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from global_utils import dump_results, get_num_true_clusters, sample_log_labels
from src.parsers.multinomial_mixture_vb import MultinomialMixtureVB


def get_single_eval(logs, true_assignments, n_clusters, n_labels, n_consts):
    np.random.seed()

    logs_p, true_assignments_p = permute_logs(logs, true_assignments)
    log_labels = sample_log_labels(true_assignments_p, n_labels)
    ev = Evaluator(true_assignments_p)
    oracle = Oracle(true_assignments_p)

    # Drain performance
    parser = Drain(logs_p, 4, 100, 0.5)
    parser.parse()
    c_drain = parser.cluster_templates
    score_drain = ev.get_ami(c_drain)

    # MultinomialMixtureVB performance
    mm = MultinomialMixtureVB()
    mm.fit(logs_p, n_clusters, log_labels=log_labels)
    c = mm.predict(logs_p)
    W = oracle.get_corr_constraints_matrix(
        parsed_clusters=c,
        n_constraint_samples=n_consts,
        tokenized_logs=logs_p,
        weight=1e7)
    mm.fit(logs_p, n_clusters, p_weights=W, sample_resp=False)
    c = mm.predict(logs_p)
    score_mm = ev.get_ami(c)

    return score_drain, score_mm


def run_exp1_4_dataset(data_config, n_labels, n_consts, n_samples):
    data_manager = DataManager(data_config)
    logs = data_manager.get_tokenized_raw_logs()
    # logs_reg = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()
    n_clusters = get_num_true_clusters(true_assignments)

    results = {
        'drain_score_samples': [],
        'mm_score_samples': [],
        'n_labels': n_labels,
        'n_consts': n_consts,
    }

    with mp.Pool(mp.cpu_count()) as pool:
        args = (logs, true_assignments, n_clusters, n_labels, n_consts)
        arg_list = [args for _ in range(n_samples)]
        mp_results = pool.starmap(get_single_eval, arg_list)
        score_drain_samples, score_mm_samples = list(zip(*mp_results))
        results['drain_score_samples'] = score_drain_samples
        results['mm_score_samples'] = score_mm_samples

    return results


def run_exp1_4_full(data_configs, n_labels, n_consts, n_samples):
    results = {}
    for data_config in data_configs:
        name = data_config['name']
        print(name)
        results[name] = run_exp1_4_dataset(data_config, n_labels,
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

    n_labels = 50
    n_consts = 100
    n_samples = 1000

    filename = 'exp1_4_results.p'
    results = run_exp1_4_full(data_configs, n_labels, n_consts, n_samples)
    dump_results(filename, results)

    minutes_taken = (time() - start_time) / 60
    print('\nTime taken: {:.4f} minutes'.format(minutes_taken))
