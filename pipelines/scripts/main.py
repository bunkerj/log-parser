from global_constants import NAME, FUNCTION
from src.data_config import DataConfigs
from pipelines.utils import get_results_dir_from_args
from exp.mixture_models.online_em import run_online_em
from pipelines.experiments_pipeline import ExperimentsPipeline
from exp.mixture_models.online_benchmark import run_online_benchmark
from exp.mixture_models.feedback_evaluation_mp import run_feedback_evaluation_mp

if __name__ == '__main__':
    results_dir = get_results_dir_from_args()
    jobs = [
        {
            NAME: 'run_feedback_evaluation_mp',
            FUNCTION: run_feedback_evaluation_mp,
            'data_configs': [
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
            ],
            'n_samples': 50,
            'label_count_values': [0, 20, 40, 60, 80, 100],
        },
        {
            NAME: 'run_online_em',
            FUNCTION: run_online_em,
            'data_config': DataConfigs.Apache,
            'n_sample': 100,
            'training_sizes': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900,
                               1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
                               1800, 1900, 2000],
        },
        {
            NAME: 'run_online_benchmark_40_lab',
            FUNCTION: run_online_benchmark,
            'n_labels': 40,
        },
        {
            NAME: 'run_online_benchmark_80_lab',
            FUNCTION: run_online_benchmark,
            'n_labels': 80,
        }
    ]

    pipe = ExperimentsPipeline(jobs, results_dir)
    pipe.run_experiments()
    pipe.write_results(results_dir)
