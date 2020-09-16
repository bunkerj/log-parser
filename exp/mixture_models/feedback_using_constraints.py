"""
Evaluate how well the online multinomial mixture model works when an oracle is
used to provide constraints.
"""
from global_utils import dump_results
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from src.helpers.oracle import Oracle
from src.parsers.drain import Drain
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline


def run_feedback_using_constraints(data_configs, drain_parameters):
    results = {
        'drain': {},
        'mmo': {},
        'constraints': {},
    }

    for data_config in data_configs:
        name = data_config['name']
        print('{}...'.format(name))

        data_manager = DataManager(data_config)
        logs = data_manager.get_tokenized_logs()
        true_assignments = data_manager.get_true_assignments()
        evaluator = Evaluator(true_assignments)

        drain = Drain(logs, *drain_parameters[name])
        drain.parse()
        drain_clusters = drain.cluster_templates
        drain_score = evaluator.get_accuracy(drain_clusters)

        mmo = MultinomialMixtureOnline(logs,
                                       len(drain_clusters) + 20,
                                       improvement_rate=1.50,
                                       is_classification=True,
                                       epsilon=0.01,
                                       alpha=1.05,
                                       beta=1.05)
        mmo.label_logs(drain_clusters, logs)
        apply_constraints(mmo, logs, true_assignments)

        logs_1 = logs[:len(logs) // 2]
        true_assignments_1 = true_assignments[:len(logs) // 2]
        mmo.perform_online_batch_em(logs_1)
        apply_constraints(mmo, logs_1, true_assignments_1)

        logs_2 = logs[len(logs) // 2:]
        mmo.perform_online_batch_em(logs_2)

        mmo_clusters = mmo.get_clusters(logs)
        mmo_score = evaluator.get_accuracy(mmo_clusters)

        results['drain'][name] = drain_score
        results['mmo'][name] = mmo_score
        results['constraints'][name] = None

    print('Done!')
    return results


def apply_constraints(mmo, logs, true_assignments):
    mmo_clusters = mmo.get_clusters(logs)
    oracle = Oracle(true_assignments)
    constraints = oracle.get_corr_constraints(mmo_clusters, 10, logs)
    mmo.enforce_constraints(constraints)
    return constraints


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

    results = run_feedback_using_constraints(data_configs,
                                             drain_parameters)
    dump_results('feedback_using_constraints.p', results)
