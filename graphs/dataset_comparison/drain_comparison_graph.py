"""
Plots a histogram of Drain experiments (from the dataset_comparison directory) to compare the accuracies against the
ones acquired from the LogPAI open source benchmark.

In the comments, are the results as reported by the associated benchmark paper from Zhu et al., 2019.
"""

from graphs.utils import show_dataset_comparison_graph, load_results

drain_benchmark_accuracies = {      # Results from benchmark paper
    'Android': 0.911,               # 0.911
    'Apache': 1.000,                # 1.000
    'BGL': 0.963,                   # 0.963
    'Hadoop': 0.948,                # 0.948
    'HDFS': 0.998,                  # 0.998
    'HealthApp': 0.780,             # 0.780
    'HPC': 0.887,                   # 0.887
    'Linux': 0.690,                 # 0.690
    'Mac': 0.787,                   # 0.787
    'OpenSSH': 0.788,               # 0.788
    'OpenStack': 0.733,             # 0.733
    'Proxifier': 0.527,             # 0.527
    'Spark': 0.920,                 # 0.920
    'Thunderbird': 0.955,           # 0.955
    'Windows': 0.997,               # 0.997
    'Zookeeper': 0.967,             # 0.967
}

final_best_accuracies = load_results('drain_dataset_comparison.p')

show_dataset_comparison_graph('Drain Accuracies',
                              drain_benchmark_accuracies,
                              final_best_accuracies)
