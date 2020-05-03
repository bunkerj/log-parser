"""
Create separate files which contain the templates extracted from structured
files.
"""
from data.scripts.utils import extract_templates
from src.data_config import DataConfigs
from src.utils import write_csv

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
    DataConfigs.Zookeeper,
]

for data_config in data_configs:
    structured_file = data_config['assignments_path']
    templates = extract_templates(structured_file)
    csv_contents = {'EventId': ['E{}'.format(t) for t in templates],
                    'EventTemplate': list(templates.values())}
    template_path = 'data/full/templates/{}_templates.csv'.format(
        data_config['name'])
    write_csv(template_path, csv_contents)
