import os
import pickle
import numpy as np
from scipy.special import gammaln

from global_constants import RESULTS_DIR


def dump_results(name, results):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    path = os.path.join(RESULTS_DIR, name)
    pickle.dump(results, open(path, 'wb'))


def load_results(name):
    path = os.path.join(RESULTS_DIR, name)
    return pickle.load(open(path, 'rb'))


def multi(x, params):
    x_flat = x.flatten()
    params_flat = np.maximum(params, 0).flatten()
    coeff = gammaln(x_flat.sum() + 1) - np.sum(gammaln(x_flat + 1))

    valid_indices_x = np.nonzero(x_flat)
    valid_indices_params = np.nonzero(params_flat)

    if not np.all(np.isin(valid_indices_x, valid_indices_params)):
        return 0.0

    log_result = coeff + (x_flat[valid_indices_x] * np.log(
        params_flat[valid_indices_x])).sum()
    result = np.exp(log_result)

    return result
