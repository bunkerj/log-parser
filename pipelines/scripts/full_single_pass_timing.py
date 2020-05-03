from exp.mixture_models.single_pass_timing import run_single_pass_timing
from global_constants import NAME, FUNCTION
from src.data_config import DataConfigs
from pipelines.utils import get_results_dir_from_args
from pipelines.experiments_pipeline import ExperimentsPipeline

if __name__ == '__main__':
    results_dir = get_results_dir_from_args()

    data_config_zip = [
        (DataConfigs.Android, DataConfigs.Android_FULL),
        (DataConfigs.Apache, DataConfigs.Apache_FULL),
        (DataConfigs.BGL, DataConfigs.BGL_FULL),
        (DataConfigs.Hadoop, DataConfigs.Hadoop_FULL),
        (DataConfigs.HDFS, DataConfigs.HDFS_FULL),
        (DataConfigs.HealthApp, DataConfigs.HealthApp_FULL),
        (DataConfigs.HPC, DataConfigs.HPC_FULL),
        (DataConfigs.Linux, DataConfigs.Linux_FULL),
        (DataConfigs.Mac, DataConfigs.Mac_FULL),
        (DataConfigs.OpenSSH, DataConfigs.OpenSSH_FULL),
        (DataConfigs.OpenStack, DataConfigs.OpenStack_FULL),
        (DataConfigs.Proxifier, DataConfigs.Proxifier_FULL),
        (DataConfigs.Spark, DataConfigs.Spark_FULL),
        (DataConfigs.Zookeeper, DataConfigs.Zookeeper_FULL),
    ]

    jobs = []
    for data_config_tuple in data_config_zip:
        name = data_config_tuple[0]['name'].lower()
        job = {
            NAME: 'single_pass_timing_{}'.format(name),
            FUNCTION: run_single_pass_timing,
            'data_config': data_config_tuple[1],
            'init_data_config': data_config_tuple[0],
            'limit': 50000,
        }
        jobs.append(job)

    pipe = ExperimentsPipeline(jobs)
    pipe.run_experiments()
    pipe.write_results(results_dir)
