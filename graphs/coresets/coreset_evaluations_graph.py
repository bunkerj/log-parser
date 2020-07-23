import matplotlib.pyplot as plt
from global_utils import load_results
from graphs.utils import get_num_unique_log_mapping
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
]

labels = []
means = []
num_unique_log_mapping = get_num_unique_log_mapping()

for idx, data_config in enumerate(data_configs, start=1):
    plt.subplot(*SUBPLOT_DIM, idx)
    name = data_config['name'].lower()
    n_unique_logs = num_unique_log_mapping[name]
    results = load_results(
        'coreset_evaluations_{}_nmi_pos.p'.format(name))
    score_samples_offline = results['score_samples_offline']
    score_samples_online = results['score_samples_online']
    score_samples_coreset = results['score_samples_coreset']
    coreset_size = results['coreset_size']

    plt.title(
        '{} ($n_c={}$, $n_u={}$)'.format(name, coreset_size, n_unique_logs))
    plt.boxplot([score_samples_offline,
                 score_samples_coreset,
                 score_samples_online],
                labels=['offline', 'coreset', 'online'])
    plt.grid()

plt.show()
