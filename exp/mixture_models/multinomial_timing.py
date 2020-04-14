import numpy as np
from time import time
from scipy.stats import multinomial as multi_scipy
from global_utils import multi as multi_custom_pmf

N_SAMPLES = 500000
N_CLASSES = 200
N_EXPERIMENTS = 1000

total_params = np.random.dirichlet(np.ones(N_CLASSES), size=N_SAMPLES)
samples = np.zeros((N_SAMPLES, N_CLASSES))
for idx, params in enumerate(total_params):
    samples[idx, :] = np.random.multinomial(N_EXPERIMENTS, params)

scipy_multi_time = time()
for idx in range(N_SAMPLES):
    result = multi_custom_pmf(samples[idx, :], total_params[idx, :])
scipy_multi_time = time() - scipy_multi_time

custom_multi_time = time()
for idx in range(N_SAMPLES):
    multi_scipy.pmf(samples[idx, :], sum(samples[idx, :]), total_params[idx, :])
custom_multi_time = time() - custom_multi_time

print('Scipy implementation: {}'.format(scipy_multi_time))
print('Custom implementation: {}'.format(custom_multi_time))
