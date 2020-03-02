"""
For the specified datasets, plot the impurity as a function of labeled data
points.
"""
from utils import load_results
import matplotlib.pyplot as plt

from src.data_config import DataConfigs

data_configs = [
    DataConfigs.Android,
    DataConfigs.Apache,
    DataConfigs.BGL,
    DataConfigs.Hadoop,
    DataConfigs.HDFS,
    DataConfigs.HealthApp,
    DataConfigs.HPC,
    DataConfigs.Linux,
    DataConfigs.Mac,
    DataConfigs.OpenSSH,
    DataConfigs.OpenStack,
    DataConfigs.Proxifier,
    DataConfigs.Spark,
    DataConfigs.Thunderbird,
    DataConfigs.Windows,
    DataConfigs.Zookeeper,
]

for idx, data_config in enumerate(data_configs):
    dataset_name = data_config['name']

    results = load_results(
        'feedback_evaluation_mp_filtered_no_num_{}_10s.p'.format(
            dataset_name.lower()))

    label_counts = results['label_counts']
    labeled_impurities = results['avg_labeled_impurities']
    unlabeled_impurities = results['avg_unlabeled_impurities']

    ax = plt.subplot(4, 4, idx + 1)
    plt.plot(label_counts, labeled_impurities)
    plt.plot(label_counts, unlabeled_impurities)
    ax.text(.5, .9, dataset_name,
            horizontalalignment='center',
            transform=ax.transAxes)
    if idx == 3:
        plt.legend(['Labeled', 'Unlabeled'])
    plt.grid()

plt.show()
