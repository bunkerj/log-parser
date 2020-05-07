import os
import pickle
import numpy as np
from random import shuffle
from scipy.special import gammaln
from global_constants import RESULTS_DIR


def dump_results(name, results, results_dir=None):
    results_dir = RESULTS_DIR if results_dir is None else results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    path = os.path.join(results_dir, name)
    pickle.dump(results, open(path, 'wb'))


def load_results(name, results_dir=RESULTS_DIR):
    path = os.path.join(results_dir, name)
    return pickle.load(open(path, 'rb'))


def shuffle_same_order(*arrays):
    c = list(zip(*arrays))
    shuffle(c)
    return list(zip(*c))


def log_multi(x, params):
    x_flat = x.flatten()
    params_flat = np.maximum(params, 0).flatten()
    coeff = gammaln(x_flat.sum() + 1) - np.sum(gammaln(x_flat + 1))

    valid_indices_x = np.nonzero(x_flat)
    valid_indices_params = np.nonzero(params_flat)

    if not np.all(np.isin(valid_indices_x, valid_indices_params)):
        return -np.inf

    return coeff + (x_flat[valid_indices_x] * np.log(
        params_flat[valid_indices_x])).sum()


def multi(x, params):
    return np.exp(log_multi(x, params))
