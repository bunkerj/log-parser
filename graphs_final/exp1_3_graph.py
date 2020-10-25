import matplotlib.pyplot as plt
from statistics import mean
from global_utils import load_results
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager

DIM = (4, 4)

results = load_results('exp1_3_results_new.p')

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

    ds_results = results[name]

    drain_score = ds_results['drain_score']
    enhanced_score = mean(ds_results['enhanced_score'])
    ub = 1
    lb = min(drain_score, enhanced_score) - 0.05

    plt.subplot(*DIM, idx)
    plt.title(name)
    plt.bar(['Drain', 'Enhanced'], [drain_score, enhanced_score])
    plt.ylim([lb, ub])
    plt.grid()

plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.show()
