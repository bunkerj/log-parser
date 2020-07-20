import numpy as np
from time import time
from global_utils import multi, get_multi_values
from scipy.special import gammaln
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from exp.mixture_models.utils import get_num_true_clusters
from src.utils import get_vocabulary_indices, get_token_counts


def log_multi_ref(x, params):
    x_flat = x.flatten()
    params_flat = np.maximum(params, 0).flatten()
    coeff = gammaln(x_flat.sum() + 1) - np.sum(gammaln(x_flat + 1))
    return coeff + (x_flat * np.log(params_flat)).sum()


def multi_ref(x, params):
    return np.exp(log_multi_ref(x, params))


def run_sparse_multinomial_timing(data_config):
    data_manager = DataManager(data_config)
    tokenized_logs = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()

    num_clusters = get_num_true_clusters(true_assignments)
    v_indices = get_vocabulary_indices(tokenized_logs)
    num_vocab = len(v_indices)
    vocab_pos = np.ones((num_clusters, num_vocab))
    count_vectors = [get_token_counts(log, v_indices) for log in tokenized_logs]
    theta = np.vstack([np.random.dirichlet(vp) for vp in vocab_pos])

    start_time = time()
    reference_sum = 0
    for x_n in count_vectors:
        for g in range(num_clusters):
            reference_sum += multi(x_n, theta[g, :])
    ref_time = time() - start_time

    start_time = time()
    candidate_sum = 0
    for x_n in count_vectors:
        multi_values = get_multi_values(x_n, theta)
        candidate_sum += multi_values.sum()
    candidate_time = time() - start_time

    return {
        'ref_time': ref_time,
        'candidate_time': candidate_time,
        'sum_diff': abs(reference_sum - candidate_sum)
    }


if __name__ == '__main__':
    data_config = DataConfigs.BGL

    results = run_sparse_multinomial_timing(data_config)
    for k, v in results.items():
        print('{:<20}{:<10.10}'.format(k, v))
