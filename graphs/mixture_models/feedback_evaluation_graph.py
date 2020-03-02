"""
For the specified datasets, plot the impurity as a function of labeled data
points.
"""
import matplotlib.pyplot as plt
from global_utils import load_results
from src.data_config import DataConfigs

N_SAMPLES = 3

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
    name = data_config['name']

    results = load_results('feedback_eval_{}_{}s.p'.format(name.lower(),
                                                           N_SAMPLES))

    label_counts = results[name]['label_counts']
    labeled_impurities = results[name]['avg_labeled_impurities']
    unlabeled_impurities = results[name]['avg_unlabeled_impurities']

    ax = plt.subplot(4, 4, idx + 1)
    plt.plot(label_counts, labeled_impurities)
    plt.plot(label_counts, unlabeled_impurities)
    ax.text(.5, .9, name, horizontalalignment='center', transform=ax.transAxes)
    if idx == 3:
        plt.legend(['Labeled', 'Unlabeled'])
    plt.grid()

plt.show()
