"""
For the specified dataset, plot the mean impurity as a function of labeled data
points. Also plot the samples in lighter colors alongside the means.
"""
import matplotlib.pyplot as plt
from global_utils import load_results
from src.data_config import DataConfigs
from global_constants import N_LOGS, LABEL_COUNTS, LABELED_IMPURITIES_SAMPLES, \
    UNLABELED_IMPURITIES_SAMPLES, AVG_LABELED_IMPURITIES, \
    AVG_UNLABELED_IMPURITIES

DATA_CONFIG = DataConfigs.Apache
name = DATA_CONFIG['name']
results = load_results('feedback_evaluation_mp.p')
dataset_results = results[name]

n_logs = dataset_results[N_LOGS]
label_counts = dataset_results[LABEL_COUNTS]
lab_samples = dataset_results[LABELED_IMPURITIES_SAMPLES]
unlab_samples = dataset_results[UNLABELED_IMPURITIES_SAMPLES]
labeled_impurities = dataset_results[AVG_LABELED_IMPURITIES]
unlabeled_impurities = dataset_results[AVG_UNLABELED_IMPURITIES]

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
