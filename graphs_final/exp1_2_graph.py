import matplotlib.pyplot as plt
from statistics import mean, stdev
from global_utils import load_results
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from graphs_final.utils import get_sample_avg, get_sample_std, \
    plot_mean_with_std

DIM = (4, 4)
FEEDBACK_TYPE = 'labels'
# FEEDBACK_TYPE = 'constraints'

results = load_results('exp1_2_results.p')

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
    feedback_type = FEEDBACK_TYPE.lower().capitalize()
    print(name)

    data_manager = DataManager(data_config)
    logs = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()
    ev = Evaluator(true_assignments)

    ds_results = results[name]

    feedback_count = ds_results['feedback_counts'][FEEDBACK_TYPE]
    mean_base_val = mean(ds_results['samples']['base'])
    std_base_val = stdev(ds_results['samples']['base'])
    mean_base = [mean_base_val for _ in range(len(feedback_count))]
    std_base = [std_base_val for _ in range(len(feedback_count))]

    mean_feedback = get_sample_avg(ds_results['samples'][FEEDBACK_TYPE])
    std_feedback = get_sample_std(ds_results['samples'][FEEDBACK_TYPE])

    plt.subplot(*DIM, idx)
    plt.title(name)
    plot_mean_with_std(feedback_count, mean_base, std_base, 'blue',
                       label='No Feedback')
    plot_mean_with_std(feedback_count, mean_feedback, std_feedback, 'green',
                       label=feedback_type)
    if idx == 1:
        plt.legend(loc='upper left')
        plt.ylabel('AMI')
        plt.xlabel(feedback_type)
    # plt.tight_layout(w_pad=3.0)
    plt.grid()

plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.show()
