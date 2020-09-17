import matplotlib.pyplot as plt
from global_utils import load_results, get_labeled_indices
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from statistics import mean, stdev

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
    print(name)

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
        log_labels_sample = log_labels_samples[sample_idx]
        labeled_indices = get_labeled_indices(log_labels_sample)

        score_base = ev.get_ami(c_base_sample, labeled_indices)
        score_lab = ev.get_ami(c_lab_sample, labeled_indices)
        score_lob_const = ev.get_ami(c_lab_const_sample, labeled_indices)

        scores_base.append(score_base)
        scores_lab.append(score_lab)
        scores_lab_const.append(score_lob_const)

    avg_score_base = mean(scores_base)
    avg_score_lab = mean(scores_lab)
    avg_score_lab_const = mean(scores_lab_const)

    std_score_base = stdev(scores_base)
    std_score_lab = stdev(scores_lab)
    std_score_lab_const = stdev(scores_lab_const)
    y_axis_ub = max(avg_score_base + std_score_base,
                    avg_score_lab + std_score_lab,
                    avg_score_lab_const + std_score_lab_const)
    y_axis_lb = min(avg_score_base - std_score_base,
                    avg_score_lab - std_score_lab,
                    avg_score_lab_const - std_score_lab_const)

    plt.subplot(*DIM, idx)
    plt.title(name)
    plt.bar(['Base',
             'Lab',
             'Lab + Const'],
            [avg_score_base,
             avg_score_lab,
             avg_score_lab_const])
    plt.ylabel('AMI')
    plt.ylim([y_axis_lb, y_axis_ub])
    plt.grid()

plt.subplots_adjust(wspace=0.3)
plt.show()
