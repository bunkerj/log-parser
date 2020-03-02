import os
import pickle

from constants import RESULTS_DIR


def dump_results(name, results):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    path = os.path.join(RESULTS_DIR, name)
    pickle.dump(results, open(path, 'wb'))


def load_results(name):
    path = os.path.join(RESULTS_DIR, name)
    return pickle.load(open(path, 'rb'))
