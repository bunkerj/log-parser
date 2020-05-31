from global_constants import NAME, FUNCTION
from src.data_config import DataConfigs
from pipelines.utils import get_results_dir_from_args
from pipelines.experiments_pipeline import ExperimentsPipeline
from exp.constraints_feedback.feedback_convergence import \
    run_feedback_convergence

if __name__ == '__main__':
    results_dir = get_results_dir_from_args()

    data_configs = [
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
    ]

    drain_parameters = {
        'Android': (5, 100, 0.21),
        'Apache': (11, 100, 0.76),
        'BGL': (5, 100, 0.54),
        'Hadoop': (3, 100, 0.66),
        'HDFS': (3, 100, 0.48),
        'HealthApp': (3, 100, 0.30),
        'HPC': (3, 100, 0.22),
        'Linux': (4, 100, 0.40),
        'Mac': (4, 100, 0.80),
        'OpenSSH': (4, 100, 0.71),
        'OpenStack': (3, 100, 0.80),
        'Proxifier': (50, 100, 0.62),
        'Spark': (6, 100, 0.75),
        'Thunderbird': (4, 100, 0.70),
        'Windows': (7, 100, 0.42),
        'Zookeeper': (4, 100, 0.60),
    }

    improvement_rates = [1.05, 1.50, 2.00, 5.00, 10.00]

    jobs = [
        {
            NAME: 'run_feedback_convergence',
            FUNCTION: run_feedback_convergence,
            'data_configs': data_configs,
            'drain_parameters': drain_parameters,
            'improvement_rates': improvement_rates,
            'n_cycles': 5,
            'constraint_type': None,
            'n_clusters_buffer': 0,
        },
        {
            NAME: 'run_feedback_convergence_20_buffer',
            FUNCTION: run_feedback_convergence,
            'data_configs': data_configs,
            'drain_parameters': drain_parameters,
            'improvement_rates': improvement_rates,
            'n_cycles': 5,
            'constraint_type': None,
            'n_clusters_buffer': 20,
        },
        {
            NAME: 'run_feedback_convergence_only_must_link',
            FUNCTION: run_feedback_convergence,
            'data_configs': data_configs,
            'drain_parameters': drain_parameters,
            'improvement_rates': improvement_rates,
            'n_cycles': 5,
            'constraint_type': 'must-link',
            'n_clusters_buffer': 0,
        },
        {
            NAME: 'run_feedback_convergence_only_cannot_link',
            FUNCTION: run_feedback_convergence,
            'data_configs': data_configs,
            'drain_parameters': drain_parameters,
            'improvement_rates': improvement_rates,
            'n_cycles': 5,
            'constraint_type': 'cannot-link',
            'n_clusters_buffer': 0,
        },
    ]

    pipe = ExperimentsPipeline(jobs, results_dir)
    pipe.run_experiments_mp()
    pipe.write_results(results_dir)
