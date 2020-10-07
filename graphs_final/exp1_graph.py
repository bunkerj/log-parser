import matplotlib.pyplot as plt
from global_utils import load_results
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator

DIM = (4, 4)

results = load_results('exp1_results.p')

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

    dataset_results = results[name]
    score_base_samples = dataset_results['score_base_samples']
    score_const_samples = dataset_results['score_const_samples']
    score_lab_samples = dataset_results['score_lab_samples']
    score_lob_const_samples \
        = dataset_results['score_lab_const_samples']

    plt.subplot(*DIM, idx)
    plt.title(name)
    plt.boxplot([score_base_samples,
                 score_const_samples,
                 score_lab_samples,
                 score_lob_const_samples],
                labels=['Base', 'C', 'L', 'L/C'],
                showfliers=False)
    plt.ylabel('AMI')
    plt.grid()

plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.show()
