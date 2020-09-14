"""
On using the 50k log dataset from BGL and 25k logs from Linux, we would perform
a run and compare the use of the coreset to choose logs that should be labeled
and compare performance against the heuristic where logs are uniformly sampled.
This would be done for different numbers of labels.
"""
import multiprocessing as mp
from time import time
from exp.mixture_models.utils import get_log_labels, get_num_true_clusters
from exp_final.utils import get_log_sample, get_coreset
from global_utils import dump_results
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from src.parsers.multinomial_mixture_vb import MultinomialMixtureVB


def run_exp4_full(data_config, label_counts, cs_proj_size,
                  subset_size, n_samples):
    results = {
        'random_label_scores': [],
        'coreset_label_scores': [],
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
            results['random_label_scores'].append(mp_results)

    # Coreset labels
    with mp.Pool(mp.cpu_count()) as pool:
        for n_labels in label_counts:
            _, cs_logs, _ \
                = get_coreset(logs, n_clusters, n_labels, cs_proj_size)
            results['cs_size'].append(len(cs_logs))

            args = (logs, n_clusters, log_labels, ev)
            arg_list = [args for _ in range(n_samples)]
            mp_results = pool.starmap(run_exp4_single, arg_list)
            results['coreset_label_scores'].append(mp_results)

    return results


def run_exp4_single(logs, n_clusters, log_labels, ev):
    mm_random = MultinomialMixtureVB()
    mm_random.fit(logs, n_clusters, log_labels=log_labels)
    c_cs = mm_random.predict(logs)
    score = ev.get_nmi(c_cs)
    return score


if __name__ == '__main__':
    start_time = time()

    data_config = DataConfigs.BGL_FULL_FINAL
    subset_size = 50000
    n_samples = 100
    label_counts = list(range(0, 250, 50))
    def_cs_proj_size = 1000

    filename = 'exp4_results.p'
    results = run_exp4_full(data_config, label_counts, def_cs_proj_size,
                            subset_size, n_samples)
    dump_results(filename, results)

    minutes_taken = (time() - start_time) / 60
    print('\nTime taken: {:.4f} minutes'.format(minutes_taken))
