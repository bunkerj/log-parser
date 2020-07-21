"""
Compute the scores as a function of the number of geodesic iterations over the
specified datasets.
"""
from global_utils import dump_results
from src.data_config import DataConfigs
from exp.coresets.utils import run_coreset_exp

if __name__ == '__main__':
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

    results = {}
    for data_config in data_configs:
        name = data_config['name'].lower()
        results[name] = {}
        for n in range(10, 101, 10):
            results[name][n] = run_coreset_exp(proj_dim=250,
                                               subset_size=n,
                                               n_samples=25,
                                               data_config=data_config)

    filename = 'coreset_performance_vs_size.p'
    dump_results(filename, results)
