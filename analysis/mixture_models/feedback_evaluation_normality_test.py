import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from utils import load_results
from src.data_config import DataConfigs
from scipy.stats import shapiro

N_SAMPLES = 100
LABEL_COUNT_IDX = 0
DATA_CONFIG = DataConfigs.Apache

dataset_name = DATA_CONFIG['name']
results = load_results(
    'feedback_evaluation_mp_filtered_no_num_{}_{}s.p'.format(
        dataset_name.lower(), N_SAMPLES))

label_counts = results['label_counts']
unlab_samples = results['unlabeled_impurities_samples']
unlab_sample_impurity_values = \
    np.array([sample[LABEL_COUNT_IDX] for sample in unlab_samples])

title = '{} Unlabeled Impurities with {} Labels and {} Samples' \
    .format(dataset_name.capitalize(), label_counts[LABEL_COUNT_IDX], N_SAMPLES)

_, p = shapiro(unlab_sample_impurity_values)

print('Shapiro-Wilk test p-value: {}'.format(p))

sm.qqplot(unlab_sample_impurity_values, line='s')
plt.title(title)
plt.grid()
plt.show()
