import matplotlib.pyplot as plt
from statistics import mean
from global_utils import load_results
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager

DIM = (4, 4)

results = load_results('exp1_4_results.p')

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

for idx, data_config in enumerate(data_configs, start=1):
    name = data_config['name']
    print(name)

    data_manager = DataManager(data_config)
    logs = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()
    ev = Evaluator(true_assignments)

    drain_score_samples = results[name]['drain_score_samples']
    mm_score_samples = results[name]['mm_score_samples']

    color = 'lightgreen' \
        if mean(mm_score_samples) > mean(drain_score_samples) \
        else 'mistyrose'

    plt.subplot(*DIM, idx, facecolor=color)
    plt.title(name)
    plt.boxplot([drain_score_samples,
                 mm_score_samples],
                labels=['Drain', 'MultiVB'],
                showfliers=False)
    plt.ylabel('AMI')
    plt.grid()

plt.subplots_adjust(left=0.05, bottom=0.06, right=0.99,
                    top=0.96, wspace=0.3, hspace=0.5)
plt.show()
