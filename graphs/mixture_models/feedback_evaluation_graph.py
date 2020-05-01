"""
For the specified datasets, plot the impurity as a function of labeled data
points.
"""
import matplotlib.pyplot as plt
from global_utils import load_results
from src.data_config import DataConfigs
from global_constants import LABEL_COUNTS, AVG_LABELED_IMPURITIES, \
    AVG_UNLABELED_IMPURITIES

data_configs = [
    DataConfigs.Apache,
    DataConfigs.Proxifier,
]

results = load_results('feedback_evaluation_mp.p')

for idx, data_config in enumerate(data_configs):
    name = data_config['name']

    dataset_results = results[name]
    label_counts = dataset_results[LABEL_COUNTS]
    labeled_impurities = dataset_results[AVG_LABELED_IMPURITIES]
    unlabeled_impurities = dataset_results[AVG_UNLABELED_IMPURITIES]

    ax = plt.subplot(1, 2, idx + 1)
    plt.plot(label_counts, labeled_impurities)
    plt.plot(label_counts, unlabeled_impurities)
    ax.text(.5, .9, name, horizontalalignment='center', transform=ax.transAxes)
    if idx == 1:
        plt.legend(['Labeled', 'Unlabeled'])
    plt.grid()

plt.show()
