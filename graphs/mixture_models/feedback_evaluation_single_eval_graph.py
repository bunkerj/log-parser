"""
For the specified dataset, plot the mean impurity as a function of labeled data
points. Also plot the samples in lighter colors alongside the means.
"""
import matplotlib.pyplot as plt
from global_utils import load_results
from src.data_config import DataConfigs

N_SAMPLES = 3
DATA_CONFIG = DataConfigs.Apache

name = DATA_CONFIG['name']
results = load_results('feedback_eval_{}_{}s.p'.format(name.lower(), N_SAMPLES))

n_logs = results['n_logs']
label_counts = results['label_counts']
lab_samples = results['labeled_impurities_samples']
unlab_samples = results['unlabeled_impurities_samples']
labeled_impurities = results['avg_labeled_impurities']
unlabeled_impurities = results['avg_unlabeled_impurities']

label_percentages = [count / n_logs for count in label_counts]

for sample in lab_samples:
    plt.plot(label_percentages, sample, color='pink')
for sample in unlab_samples:
    plt.plot(label_percentages, sample, color='lightblue')

plt.plot(label_percentages, unlabeled_impurities,
         label='Unlabeled Mean', color='blue')
plt.plot(label_percentages, labeled_impurities,
         label='Labeled Mean', color='red')

plt.title(name.capitalize())
plt.xlabel('Label Percentage')
plt.ylabel('Normalized Impurity')
plt.legend()
plt.grid()
plt.show()
