"""
Creates the template assignment for a specific dataset configuration.
"""
from src.helpers.template_assigner import TemplateAssigner
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

for data_config in data_configs:
    print(data_config['name'])
    template_assigner = TemplateAssigner(data_config)
    template_assigner.write_assignments()
    print('Done!')
