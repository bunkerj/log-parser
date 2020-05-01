import numpy as np
import multiprocessing as mp
from global_utils import dump_results
from src.data_config import DataConfigs
from exp.mixture_models.online_em import run_online_em
from exp.mixture_models.feedback_evaluation_mp import run_feedback_evaluation_mp
from exp.master_scripts.utils import execute, query_results_dir

RESULTS_DIR = query_results_dir()

if __name__ == '__main__':
    result_names = [
        'run_feedback_evaluation_mp_main.p',
        'run_online_em_main.p',
    ]

    # Initialize with *_mp experiments here since daemonic
    # processes are not allowed to have children.
    mp_results = [
        run_feedback_evaluation_mp(
            data_configs=[DataConfigs.Apache,
                          DataConfigs.Proxifier],
            n_samples=3,
            label_count_values=list(range(0, 301, 100)),
        ),
    ]

    # All non *_mp experiments go here.
    non_mp_jobs = {
        run_online_em: {
            'data_config': DataConfigs.Apache,
            'n_sample': 3,
            'training_sizes': list(np.linspace(30, 2000, 4, dtype=np.int32)),
        },
    }

    with mp.Pool(mp.cpu_count()) as pool:
        mp_results.extend(pool.starmap(execute, non_mp_jobs.items()))

    assert len(result_names) == len(mp_results)
    for idx in range(len(result_names)):
        dump_results(result_names[idx], mp_results[idx], RESULTS_DIR)
