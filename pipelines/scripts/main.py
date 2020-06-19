from global_constants import NAME, FUNCTION
from src.data_config import DataConfigs
from pipelines.utils import get_results_dir_from_args
from exp.mixture_models.online_em import run_online_em
from pipelines.experiments_pipeline import ExperimentsPipeline
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
            'n_samples': 1000,
            'label_count_values': [0, 20, 40, 60, 80, 100],
        },
        {
            NAME: 'run_online_em',
            FUNCTION: run_online_em,
            'data_config': DataConfigs.BGL,
            'n_sample': 1000,
            'training_sizes': [200, 400, 600, 800, 1000,
                               1200, 1400, 1600, 1800, 2000],
        },
    ]

    pipe = ExperimentsPipeline(jobs, results_dir)
    pipe.run_experiments_mp()
    pipe.write_results(results_dir)
