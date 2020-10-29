from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager

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

n_total_logs = 0
n_total_tokens = 0

for data_config in data_configs:
    name = data_config['name']
    data_manager = DataManager(data_config)
    logs = data_manager.get_tokenized_logs()
    n_logs = len(logs)

    n_total_logs += n_logs
    for log in logs:
        n_total_tokens += len(log)

print('Total number of logs: {}'.format(n_total_logs))
print('Total number of tokens: {}'.format(n_total_tokens))
print('Avg number of tokens per log: {}'.format(n_total_tokens / n_total_logs))
