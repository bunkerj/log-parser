"""
Using 50k logs from the BGL dataset, we want to compare how offline VB performs
vs coreset + offline VB. We want to see how performance compares against the
baseline as a function of coreset upper bound size and coreset projection size.
Both the coreset upper bound would start at 100 and increase by increments of
100 up to 1000 and projection dimension would start at 100 and increase by
increments of 200 up to 2400.
"""
import multiprocessing as mp
from time import time
from exp.mixture_models.utils import get_num_true_clusters
from exp_final.utils import get_coreset, get_log_sample
from global_utils import dump_results, load_results
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.parsers.multinomial_mixture_vb import MultinomialMixtureVB
from src.helpers.evaluator import Evaluator


def run_exp2_full(data_config, cs_ub_sizes, cs_proj_sizes, subset_size,
                  def_cs_ub_size, def_cs_proj_size, n_samples):
    results = {}

    data_manager = DataManager(data_config)
    logs, true_assignments = get_log_sample(data_manager, subset_size)
    n_clusters = get_num_true_clusters(true_assignments)
    ev = Evaluator(true_assignments)

    # Vary coreset upperbound size
    with mp.Pool(mp.cpu_count()) as pool:
        arg_list = [
            (logs, n_clusters, ev, cs_ub_size, def_cs_proj_size, n_samples) for
            cs_ub_size in cs_ub_sizes]
        mp_results = pool.starmap(run_exp2_single, arg_list)

    preds = list(zip(*mp_results))
    results['cs_ub_sizes'] = {'c_base': preds[0], 'c_cs': preds[1]}

    print('Stage 1 complete...')

    # Vary coreset projection dimensions
    with mp.Pool(mp.cpu_count()) as pool:
        arg_list = [
            (logs, n_clusters, ev, def_cs_ub_size, cs_proj_size, n_samples) for
            cs_proj_size in cs_proj_sizes]
        mp_results = pool.starmap(run_exp2_single, arg_list)

    preds = list(zip(*mp_results))
    results['cs_proj_sizes'] = {'c_base': preds[0], 'c_cs': preds[1]}

    print('Stage 2 complete...')

    return results


def run_exp2_single(logs, n_clusters, ev, sub_size, proj_dim, n_samples):
    scores_base = []
    scores_cs = []

    for _ in range(n_samples):
        mm = MultinomialMixtureVB()
        mm.fit(logs, n_clusters)
        c_base = mm.predict(logs)
        scores_base.append(ev.get_nmi(c_base))

        cs_weights, cs_logs = get_coreset(logs, n_clusters, sub_size, proj_dim)
        mm_cs = MultinomialMixtureVB()
        mm_cs.fit(cs_logs, n_clusters, cs_weights=cs_weights)
        c_cs = mm_cs.predict(logs)
        scores_cs.append(ev.get_nmi(c_cs))

    return scores_base, scores_cs


if __name__ == '__main__':
    start_time = time()

    data_config = DataConfigs.BGL_FULL_FINAL
    n_samples = 5
    cs_ub_sizes = list(range(10, 101, 10))
    cs_proj_sizes = len(list(range(600, 2501, 200)))
    subset_size = 50000
    def_cs_ub_size = 25
    def_cs_proj_size = 1000

    filename = 'exp2_results.p'
    results = run_exp2_full(data_config, cs_ub_sizes, cs_proj_sizes,
                            subset_size, def_cs_ub_size,
                            def_cs_proj_size, n_samples)
    dump_results(filename, results)

    minutes_taken = (time() - start_time) / 60
    print('\nTime taken: {:.4f} minutes'.format(minutes_taken))
