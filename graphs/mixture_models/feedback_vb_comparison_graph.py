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
]

for idx, data_config in enumerate(data_configs, start=1):
    plt.subplot(*SUBPLOT_DIM, idx)
    name = data_config['name'].lower()
    filename = 'feedback_vb_comparison_{}.p'.format(name)
    results = load_results(filename)

    reg_scores = results['regular']
    feedback_scores = results['feedback']

    reg_score_mean = sum(reg_scores) / len(reg_scores)
    feedback_score_mean = sum(feedback_scores) / len(feedback_scores)

    plt.title(name)
    plt.bar(['Regular', 'Feedback'], [reg_score_mean, feedback_score_mean])
    plt.ylabel('Score')
    plt.grid()

plt.show()
