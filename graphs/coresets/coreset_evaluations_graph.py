import matplotlib.pyplot as plt
from statistics import mean
from global_utils import load_results
from src.data_config import DataConfigs

SUBPLOT_DIM = (2, 4)

data_configs = [
    DataConfigs.Android,
    DataConfigs.Apache,
    DataConfigs.BGL,
    DataConfigs.Hadoop,
    DataConfigs.HDFS,
    DataConfigs.HealthApp,
    DataConfigs.HPC,
    DataConfigs.Linux,
    # DataConfigs.Mac,
    # DataConfigs.OpenSSH,
    # DataConfigs.OpenStack,
    # DataConfigs.Proxifier,
    # DataConfigs.Spark,
    # DataConfigs.Thunderbird,
    # DataConfigs.Windows,
    # DataConfigs.Zookeeper,
]

labels = []
means = []
for idx, data_config in enumerate(data_configs, start=1):
    plt.subplot(*SUBPLOT_DIM, idx)
    name = data_config['name'].lower()
    results = load_results('coreset_evaluations_{}.p'.format(name))
    score_samples = results['score_samples']
    coreset_score_samples = results['coreset_score_samples']
    coreset_size = results['coreset_size']

    plt.title('{} ($n_c={}$)'.format(name, coreset_size))
    plt.boxplot([score_samples, coreset_score_samples],
                labels=['score_mean', 'coreset_score_mean'])
    plt.grid()

plt.show()
