import pickle
from graphs.utils import print_dataset_comparison

iplom_benchmark_baseline_accuracies = {
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

print_dataset_comparison(iplom_benchmark_baseline_accuracies, final_best_accuracies)