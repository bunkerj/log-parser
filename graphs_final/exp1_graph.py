import matplotlib.pyplot as plt
from global_utils import load_results
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from statistics import mean

DIM = (2, 4)

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
]

for idx, data_config in enumerate(data_configs, start=1):
    name = data_config['name']
    data_manager = DataManager(data_config)
    logs = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()
    ev = Evaluator(true_assignments)

    dataset_results = results[name]
    c_base_samples = dataset_results['clustering_base_samples']
    c_lab_samples = dataset_results['clustering_lab_samples']
    c_lab_const_samples \
        = dataset_results['clustering_lab_const_samples']
    log_labels_samples = dataset_results['log_labels_samples']

    n_samples = len(c_base_samples)
    scores_base = []
    scores_lab = []
    scores_lab_const = []
    for sample_idx in range(n_samples):
        c_base_sample = c_base_samples[sample_idx]
        c_lab_sample = c_lab_samples[sample_idx]
        c_lab_const_sample = c_lab_const_samples[sample_idx]
        log_labels_samples = log_labels_samples[sample_idx]
        scores_base.append(ev.get_ami(c_base_sample, log_labels_samples))
        scores_lab.append(ev.get_ami(c_lab_sample, log_labels_samples))
        scores_lab_const.append(ev.get_ami(c_lab_const_sample,
                                           log_labels_samples))

    avg_score_base = mean(scores_base)
    avg_score_lab = mean(scores_lab)
    avg_score_lab_const = mean(scores_lab_const)

    plt.subplot(*DIM, idx)
    plt.title(name)
    plt.bar(['Base',
             'Lab',
             'Lab + Const'],
            [avg_score_base,
             avg_score_lab,
             avg_score_lab_const])
    plt.ylabel('AMI')
    plt.grid()

plt.subplots_adjust(wspace=0.3)
plt.show()
