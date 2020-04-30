import numpy as np
from time import time
from scipy.stats import multinomial as multi_scipy
from global_utils import dump_results, multi as multi_custom_pmf


def run_multinomial_timing(n_samples, n_classes, n_experiments, name):
    params = np.random.dirichlet(np.ones(n_classes), size=n_samples)
    samples = np.zeros((n_samples, n_classes))
    for idx, single_multi_params in enumerate(params):
        samples[idx, :] = np.random.multinomial(n_experiments,
                                                single_multi_params)

    custom_multi_time = time()
    for idx in range(n_samples):
        multi_custom_pmf(samples[idx, :], params[idx, :])
    custom_multi_time = time() - custom_multi_time

    scipy_multi_time = time()
    for idx in range(n_samples):
        multi_scipy.pmf(samples[idx, :], sum(samples[idx, :]), params[idx, :])
    scipy_multi_time = time() - scipy_multi_time

    results = {
        'custom_time': custom_multi_time,
        'scipy_time': scipy_multi_time,
    }

    dump_results(name, results)


if __name__ == '__main__':
    n_samples = 500000
    n_classes = 200
    n_experiments = 1000
    name = 'multinomial_timing.p'

    run_multinomial_timing(n_samples, n_classes, n_experiments, name)
