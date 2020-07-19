import matplotlib.pyplot as plt
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
    results = load_results(
        'coreset_evaluations_{}_nmi_pos.p'.format(name))
    score_samples_offline = results['score_samples_offline']
    score_samples_online = results['score_samples_offline']
    score_samples_coreset = results['score_samples_coreset']
    coreset_size = results['coreset_size']

    plt.title('{} ($n_c={}$)'.format(name, coreset_size))
    plt.boxplot([score_samples_offline,
                 score_samples_online,
                 score_samples_coreset],
                labels=['offline', 'online', 'coreset'])
    plt.grid()

plt.show()
