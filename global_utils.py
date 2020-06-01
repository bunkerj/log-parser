import os
import pickle
import numpy as np
from random import shuffle
from scipy.special import gammaln
from global_constants import RESULTS_DIR


def dump_results(name, results, results_dir=None):
    path = create_file_path(name, results_dir)
    pickle.dump(results, open(path, 'wb'))


def load_results(name, results_dir=RESULTS_DIR):
    path = os.path.join(results_dir, name)
    return pickle.load(open(path, 'rb'))


def create_file_path(name, results_dir):
    results_dir = RESULTS_DIR if results_dir is None else results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    path = os.path.join(results_dir, name)
    return path


def shuffle_same_order(*arrays):
    c = list(zip(*arrays))
    shuffle(c)
    return list(zip(*c))


def log_multi(x, params):
    x_flat = x.flatten()
    params_flat = np.maximum(params, 0).flatten()
    coeff = gammaln(x_flat.sum() + 1) - np.sum(gammaln(x_flat + 1))
    return coeff + (x_flat * np.log(params_flat)).sum()


def multi(x, params):
    return np.exp(log_multi(x, params))


def get_top_k_args(arr, k):
    return arr.flatten().argsort()[-k:][::-1]
