"""
Prints all of the templates of a given dataset that are unmatched (i.e. has a
template index of -1).
"""
from src.data_config import DataConfigs

data_configs = [
    DataConfigs.Android_FULL,
    DataConfigs.Apache_FULL,
    DataConfigs.BGL_FULL,
    DataConfigs.Hadoop_FULL,
    DataConfigs.HDFS_FULL,
    DataConfigs.HealthApp_FULL,
    DataConfigs.HPC_FULL,
    DataConfigs.Linux_FULL,
    DataConfigs.Mac_FULL,
    DataConfigs.OpenSSH_FULL,
    DataConfigs.OpenStack_FULL,
    DataConfigs.Proxifier_FULL,
    DataConfigs.Spark_FULL,
    DataConfigs.Zookeeper_FULL,
]

print('{:<15}{:<15}{:<15}'.format('Name', '# Logs', '% Match'))
for data_config in data_configs:
    n = 0
    mismatch_count = 0
    with open(data_config['assignments_path'], encoding='utf-8') as f:
        next(f)
        for line in f:
            _, event_idx = line.split(',')
            event_idx = event_idx.strip()
            if event_idx == '-1':
                mismatch_count += 1
            n += 1

    percent_match = 100 * (n - mismatch_count) / n
    print('{:<15}{:<15}{:<15.4}'.format(data_config['name'], n, percent_match))
