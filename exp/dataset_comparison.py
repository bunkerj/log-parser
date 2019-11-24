from src.methods.iplom import Iplom
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator

data_set_configs = [
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

for data_set_config in data_set_configs:
    iplom = Iplom(data_set_config, **{
        'file_threshold': 0.0,
        'partition_threshold': 0.0,
        'lower_bound': 0.1,
        'upper_bound': 0.9,
        'goodness_threshold': 0.35,
    })
    iplom.parse()
    evaluator = Evaluator(data_set_config, iplom.cluster_templates)
    iplom_bgl_accuracy = evaluator.evaluate()
    print('Final IPLoM {} Accuracy: {}'.format(data_set_config['name'], iplom_bgl_accuracy))
