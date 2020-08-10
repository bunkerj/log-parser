import matplotlib.pyplot as plt
from global_utils import load_results
from src.data_config import DataConfigs
from statistics import mean

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

    scores = results['base']
    scores_lab = results['labeled']
    scores_lab_cont = results['labeled_const']

    mean_score = mean(scores)
    mean_score_lab = mean(scores_lab)
    mean_score_lab_cont = mean(scores_lab_cont)

    plt.title(name)
    plt.bar(['Base', 'Lab', 'Lab + Const'],
            [mean_score, mean_score_lab, mean_score_lab_cont])
    plt.grid()

plt.show()
