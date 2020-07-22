"""
Compute the scores as a function of projection dimension over the specified
datasets.
"""
from global_utils import dump_results
from src.data_config import DataConfigs
from exp.coresets.utils import run_coreset_exp_mp

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
        print(name)
        results[name] = {}
        for dim in range(10, 600, 50):
            results[name][dim] = run_coreset_exp_mp(proj_dim=dim,
                                                    subset_size=50,
                                                    n_samples=100,
                                                    data_config=data_config)

    filename = 'coreset_performance_vs_dim.p'
    dump_results(filename, results)
