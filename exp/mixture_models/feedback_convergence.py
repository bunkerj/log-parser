from global_utils import dump_results
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from src.helpers.oracle import Oracle
from src.parsers.drain import Drain
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline


def run_feedback_convergence(data_configs, drain_parameters,
                             improvement_rates, n_runs):
    results = {}
    for data_config in data_configs:
        name = data_config['name']
        print('{}...'.format(name))
        results[name] = {}

        data_manager = DataManager(data_config)
        logs = data_manager.get_tokenized_logs()
        true_assignments = data_manager.get_true_assignments()
        ev = Evaluator(true_assignments)

        drain = Drain(logs, *drain_parameters[name])
        drain.parse()
        drain_clusters = drain.cluster_templates

        for improvement_rate in improvement_rates:
            print('Rate: {}'.format(improvement_rate))

            results[name][improvement_rate] = {}
            mmo = MultinomialMixtureOnline(logs,
                                           len(drain_clusters),
                                           improvement_rate=improvement_rate,
                                           is_classification=True,
                                           epsilon=0.01,
                                           alpha=1.05,
                                           beta=1.05)
            mmo.label_logs(drain_clusters, logs)

            oracle = Oracle(true_assignments)
            acc_vals, t1_vals, t2_vals = perform_runs(ev, logs, mmo,
                                                      n_runs, oracle)
            results[name][improvement_rate]['acc'] = acc_vals
            results[name][improvement_rate]['t1'] = t1_vals
            results[name][improvement_rate]['t2'] = t2_vals

    return results


def perform_runs(ev, logs, mmo, n_runs, oracle):
    acc_vals, t1_vals, t2_vals = [], [], []
    for _ in range(n_runs):
        clusters = mmo.get_clusters(logs)
        constraints = oracle.get_constraints(clusters, 1, logs)
        mmo.enforce_constraints(constraints)

        acc_val = ev.get_accuracy(clusters)
        t1_val = ev.get_type1_error_ratio()
        t2_val = ev.get_type2_error_ratio()

        acc_vals.append(acc_val)
        t1_vals.append(t1_val)
        t2_vals.append(t2_val)

    return acc_vals, t1_vals, t2_vals


if __name__ == '__main__':
    n_runs = 20

    improvement_rates = [1.05, 1.50, 2.00]

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

    results = run_feedback_convergence(data_configs,
                                       drain_parameters,
                                       improvement_rates,
                                       n_runs)
    dump_results('feedback_convergence.p', results)
