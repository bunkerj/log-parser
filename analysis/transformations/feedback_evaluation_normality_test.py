"""
Perform the Shapiro-Wilk test and plot the Q-Q plot for the impurity samples
for a given dataset at a given label percentage value.
"""
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from global_utils import load_results
from src.data_config import DataConfigs
from scipy.stats import shapiro
from global_constants import LABEL_COUNTS, UNLABELED_IMPURITIES_SAMPLES

N_SAMPLES = 3
LABEL_COUNT_IDX = 0
DATA_CONFIG = DataConfigs.Apache

name = DATA_CONFIG['name']
results = load_results('feedback_evaluation_mp.p')
dataset_results = results[name]

label_counts = dataset_results[LABEL_COUNTS]
unlab_samples = dataset_results[UNLABELED_IMPURITIES_SAMPLES]
unlab_sample_impurity_values = \
    np.array([sample[LABEL_COUNT_IDX] for sample in unlab_samples])

title = '{} Unlabeled Impurities with {} Labels and {} Samples' \
    .format(name.capitalize(), label_counts[LABEL_COUNT_IDX],
            len(unlab_samples))

_, p = shapiro(unlab_sample_impurity_values)

print('Shapiro-Wilk test p-value: {}'.format(p))

sm.qqplot(unlab_sample_impurity_values, line='s')
plt.title(title)
plt.grid()
plt.show()
