"""
Experiment that measures the amount of time it takes to do one pass over a
specific dataset.
"""
from time import time
from global_utils import dump_results
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from exp.mixture_models.utils import get_num_true_clusters
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline


def run_single_pass_timing(data_config, init_data_config, limit):
    data_manager = DataManager(init_data_config)
    initial_logs = data_manager.get_tokenized_no_num_logs()
    true_assignments = data_manager.get_true_assignments()
    n_true_clusters = get_num_true_clusters(true_assignments)

    parser = MultinomialMixtureOnline(initial_logs,
                                      n_true_clusters,
                                      is_classification=False,
                                      alpha=1.05,
                                      beta=1.05)

    with open(data_config['unstructured_path'], encoding='utf-8') as f:
        start_time = time()
        for idx, raw_log in enumerate(f, start=1):
            if idx % 5000 == 0:
                print('{}...'.format(idx))
            log = data_manager.get_tokenized_log(raw_log)
            if log is not None:
                parser.perform_online_em(log)
            if idx == limit:
                break
        timing = time() - start_time

    print('Done')

    return {
        'name': data_config['name'],
        'n_clusters': n_true_clusters,
        'n_vocab': len(parser.v_indices),
        'timing': timing,
    }


if __name__ == '__main__':
    data_config = DataConfigs.Apache_FULL
    init_data_config = DataConfigs.Apache

    results = run_single_pass_timing(data_config, init_data_config, 50000)
    dump_results('single_pass_timing.p', results)
