"""
Using 50k logs from the BGL dataset, we want to compare how offline VB performs
vs coreset + offline VB. We want to see how performance compares against the
baseline as a function of coreset upper bound size and coreset projection size.
The goal is to provide two line plots: performance as a function of upper bound
size and performance as a function of projection size.
"""
import numpy as np
import multiprocessing as mp
from time import time
from exp_final.utils import get_coreset, get_log_sample
from global_utils import dump_results, get_num_true_clusters
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.parsers.multinomial_mixture_vb import MultinomialMixtureVB
from src.helpers.evaluator import Evaluator


def run_exp2_full(data_config, cs_ub_sizes, cs_proj_sizes, subset_size,
                  def_cs_ub_size, def_cs_proj_size, n_samples):
    results = {
        'cs_ub': {'ami_base_samples': [],
                  'ami_cs_samples': [],
                  'acc_base_samples': [],
                  'acc_cs_samples': [],
                  'varying_sizes': cs_ub_sizes,
                  'cs_sizes': [],
                  },
        'cs_proj': {'ami_base_samples': [],
                    'ami_cs_samples': [],
                    'acc_base_samples': [],
                    'acc_cs_samples': [],
                    'varying_sizes': cs_proj_sizes,
                    'cs_sizes': [],
                    },
    }

    data_manager = DataManager(data_config)
    logs, true_assignments = get_log_sample(data_manager, subset_size)
    n_clusters = get_num_true_clusters(true_assignments)
    ev = Evaluator(true_assignments)

    # Vary coreset upperbound size
    with mp.Pool(mp.cpu_count()) as pool:
        for cs_ub_size in cs_ub_sizes:
            args = (logs, n_clusters, ev, cs_ub_size, def_cs_proj_size)
            arg_list = [args for _ in range(n_samples)]
            mp_results = pool.starmap(run_exp2_single, arg_list)
            ami_base_samples, acc_base_samples, ami_cs_samples, \
            acc_cs_samples, cs_sizes = list(zip(*mp_results))
            results['cs_ub']['ami_base_samples'].append(ami_base_samples)
            results['cs_ub']['acc_base_samples'].append(acc_base_samples)
            results['cs_ub']['ami_cs_samples'].append(ami_cs_samples)
            results['cs_ub']['acc_cs_samples'].append(acc_cs_samples)
            results['cs_ub']['cs_sizes'].append(cs_sizes)

    print('Stage 1 complete...')

    # # Vary coreset projection dimensions
    # with mp.Pool(mp.cpu_count()) as pool:
    #     for cs_proj_size in cs_proj_sizes:
    #         args = (logs, n_clusters, ev, def_cs_ub_size, cs_proj_size)
    #         arg_list = [args for _ in range(n_samples)]
    #         mp_results = pool.starmap(run_exp2_single, arg_list)
    #         ami_base_samples, acc_base_samples, ami_cs_samples, \
    #         acc_cs_samples, cs_sizes = list(zip(*mp_results))
    #         results['cs_proj']['ami_base_samples'].append(ami_base_samples)
    #         results['cs_proj']['acc_base_samples'].append(acc_base_samples)
    #         results['cs_proj']['ami_cs_samples'].append(ami_cs_samples)
    #         results['cs_proj']['acc_cs_samples'].append(acc_cs_samples)
    #         results['cs_proj']['cs_sizes'].append(cs_sizes)
    #
    # print('Stage 2 complete...')

    return results


def run_exp2_single(logs, n_clusters, ev, sub_size, proj_dim):
    np.random.seed()

    mm = MultinomialMixtureVB()
    mm.fit(logs, n_clusters)
    c_base = mm.predict(logs)
    ami_base = ev.get_ami(c_base)
    acc_base = ev.get_accuracy(c_base)

    cs_weights, cs_logs, _ \
        = get_coreset(logs, n_clusters, sub_size, proj_dim)
    cs_size = len(cs_logs)
    mm_cs = MultinomialMixtureVB()
    mm_cs.fit(cs_logs, n_clusters, cs_weights=cs_weights)
    c_cs = mm_cs.predict(logs)
    ami_cs = ev.get_ami(c_cs)
    acc_cs = ev.get_accuracy(c_cs)

    return ami_base, acc_base, ami_cs, acc_cs, cs_size


if __name__ == '__main__':
    start_time = time()

    data_config = DataConfigs.BGL_FULL_FINAL
    n_samples = 1000
    cs_ub_sizes = [600]
    cs_proj_sizes = list(range(100, 1001, 100))
    subset_size = 50000
    def_cs_ub_size = 50
    def_cs_proj_size = 1000

    filename = 'exp2_results.p'
    results = run_exp2_full(data_config, cs_ub_sizes, cs_proj_sizes,
                            subset_size, def_cs_ub_size,
                            def_cs_proj_size, n_samples)
    dump_results(filename, results)

    minutes_taken = (time() - start_time) / 60
    print('\nTime taken: {:.4f} minutes'.format(minutes_taken))
