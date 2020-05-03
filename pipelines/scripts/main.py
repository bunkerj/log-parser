import numpy as np
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
            'data_configs': [DataConfigs.Apache,
                             DataConfigs.Proxifier],
            'n_samples': 3,
            'label_count_values': list(range(0, 301, 100)),
        },
        {
            NAME: 'run_online_em',
            FUNCTION: run_online_em,
            'data_config': DataConfigs.Apache,
            'n_sample': 3,
            'training_sizes': list(np.linspace(30, 2000, 4, dtype=np.int32)),
        },
    ]

    pipe = ExperimentsPipeline(jobs)
    pipe.run_experiments_mp()
    pipe.write_results(results_dir)
