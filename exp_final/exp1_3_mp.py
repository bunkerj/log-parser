"""
This experiment will be run on 2k sample datasets. For each of these datasets,
we plot performance for Drain as well as our MultinomialVB initialized by Drain.
The goal is to evaluate whether MultinomialVB could be used to enhanced
deterministic approaches.
"""
import numpy as np
import multiprocessing as mp
from time import time
from src.parsers.drain import Drain
from src.helpers.oracle import Oracle
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from global_utils import dump_results, get_num_true_clusters
from src.parsers.multinomial_mixture_vb import MultinomialMixtureVB


def get_top_k_clusters(c, k):
    if k >= len(c):
        return c
    keys = sorted(c, key=lambda x: len(c[x]), reverse=True)[:k]
    top_k_clusters = {k: c[k] for k in keys}
    return top_k_clusters


def get_single_eval(logs, c_proposed, ev, oracle, n_consts):
    np.random.seed()
    n_clusters = len(c_proposed)

    mm = MultinomialMixtureVB()
    mm.fit(logs, n_clusters, log_labels=c_proposed)
    c_no_consts = mm.predict(logs)

    W = oracle.get_corr_constraints_matrix(
        parsed_clusters=c_no_consts,
        n_constraint_samples=n_consts,
        tokenized_logs=logs,
        weight=1e7)
    mm.fit(logs, n_clusters,
           log_labels=c_proposed,
           p_weights=W,
           sample_resp=False)
    c_feedback = mm.predict(logs)
    score_consts = ev.get_ami(c_feedback)

    return score_consts


def run_exp1_3_dataset(data_config, n_constraints, n_samples):
    data_manager = DataManager(data_config)
    logs = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()
    n_clusters = get_num_true_clusters(true_assignments)
    oracle = Oracle(true_assignments)
    ev = Evaluator(true_assignments)

    # Drain performance
    parser = Drain(logs, 4, 100, 0.5)
    parser.parse()
    c_drain = parser.cluster_templates
    score_drain = ev.get_ami(c_drain)
    c_proposed = get_top_k_clusters(c_drain, n_clusters)

    results = {
        'drain_score': score_drain,
        'enhanced_score': [],
        'n_constraints': n_constraints,
    }

    with mp.Pool(mp.cpu_count()) as pool:
        args = (logs, c_proposed, ev, oracle, n_constraints)
        arg_list = [args for _ in range(n_samples)]
        score_consts_samples = pool.starmap(get_single_eval, arg_list)
        results['enhanced_score'] = score_consts_samples

    return results


def run_exp1_3_full(data_configs, n_constraints, n_samples):
    results = {}
    for data_config in data_configs:
        name = data_config['name']
        print(name)
        results[name] = run_exp1_3_dataset(data_config, n_constraints,
                                           n_samples)
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

    n_constraints = 200
    n_samples = 10

    filename = 'exp1_3_results_new.p'
    results = run_exp1_3_full(data_configs, n_constraints, n_samples)
    dump_results(filename, results)

    minutes_taken = (time() - start_time) / 60
    print('\nTime taken: {:.4f} minutes'.format(minutes_taken))
