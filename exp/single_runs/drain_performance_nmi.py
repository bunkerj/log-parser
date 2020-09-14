"""
Get Drain NMI values for different datasets.
"""
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from src.parsers.drain import Drain


def run_drain_performance_nmi(data_config, parameters):
    data_manager = DataManager(data_config)
    tokenized_logs = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()
    evaluator = Evaluator(true_assignments)

    parser = Drain(tokenized_logs, **parameters)
    parser.parse()

    return evaluator.get_ami(parser.cluster_templates)


if __name__ == '__main__':
    data_configs = [
        DataConfigs.Android,
        DataConfigs.Apache,
        DataConfigs.BGL,
        DataConfigs.Hadoop,
        DataConfigs.HDFS,
        DataConfigs.HealthApp,
        DataConfigs.HPC,
        DataConfigs.Linux,
    ]

    for data_config in data_configs:
        name = data_config['name'].lower()
        parameters = {'max_depth': 50, 'max_child': 100, 'sim_threshold': 0.80}
        drain_nmi = run_drain_performance_nmi(data_config, parameters)
        print('{:<10}{:<10}'.format(name, drain_nmi))
