import pickle
import numpy as np
import matplotlib.pyplot as plt
from graphs.helpers import autolabel

benchmark_baseline_accuracies = {
    'Android': 0.712,               # 0.712
    'Apache': 1.000,                # 1.000
    'BGL': 0.939,                   # 0.939
    'Hadoop': 0.954,                # 0.954
    'HDFS': 1.000,                  # 1.000
    'HealthApp': 0.822,             # 0.822
    'HPC': 0.829,                   # 0.824
    'Linux': 0.672,                 # 0.672
    'Mac': 0.671,                   # 0.673
    'OpenSSH': 0.540,               # 0.802
    'OpenStack': 0.330,             # 0.871
    'Proxifier': 0.517,             # 0.515
    'Spark': 0.920,                 # 0.920
    'Thunderbird': 0.663,           # 0.663
    'Windows': 0.567,               # 0.567
    'Zookeeper': 0.962,             # 0.962
}

final_best_accuracies = pickle.load(open('../results/dataset_comparison.p', 'rb'))

labels = benchmark_baseline_accuracies.keys()
baseline_accuracies = benchmark_baseline_accuracies.values()
candidate_accuracies = final_best_accuracies.values()

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, baseline_accuracies, width, label='Baseline')
rects2 = ax.bar(x + width / 2, candidate_accuracies, width, label='Candidate')

ax.set_ylabel('Accuracy')
ax.set_title('IPLoM Accuracies')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

autolabel(ax, rects1)
autolabel(ax, rects2)

fig.tight_layout()

plt.show()
