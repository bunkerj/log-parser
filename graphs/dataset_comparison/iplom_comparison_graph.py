"""
Plots a histogram of IPLoM experiments (from the dataset_comparison directory) to compare the accuracies against the
ones acquired from the LogPAI open source benchmark.

In the comments, are the results as reported by the associated benchmark paper from Zhu et al., 2019.
"""

from graphs.utils import show_dataset_comparison_graph, load_results

iplom_benchmark_accuracies = {      # Results from benchmark paper
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

final_best_accuracies = load_results('iplom_dataset_comparison.p')

show_dataset_comparison_graph('IPLoM Accuracies',
                              iplom_benchmark_accuracies,
                              final_best_accuracies)
