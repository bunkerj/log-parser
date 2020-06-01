from copy import deepcopy
from global_utils import dump_results
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from src.helpers.oracle import Oracle
from src.parsers.drain import Drain
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline


class RunConfig:
    def __init__(self, data_config, local_drain_params, constraint_type,
                 n_clusters_buffer, n_cycles, n_samples, n_constraint_samples):
        data_manager = DataManager(data_config)
        true_assignments = data_manager.get_true_assignments()
        logs = data_manager.get_tokenized_logs()
        drain_clusters = self._get_drain_clusters(logs, local_drain_params)

        self.n_clusters_buffer = n_clusters_buffer
        self.logs = logs
        self.ev = Evaluator(true_assignments)
        self.oracle = Oracle(true_assignments)
        self.drain_clusters = drain_clusters
        self.constraint_type = constraint_type
        self.n_cycles = n_cycles
        self.n_samples = n_samples
        self.n_constraint_samples = n_constraint_samples
        self.improvement_rate = None
        self.mmo_base = None
        self.is_class_flag = None

    def init_base_mmo(self):
        """
        Initializes a base model separately since the labeling process requires
        the determination of whether EM or CEM is used.
        """
        n_clusters = self._get_n_clusters()
        is_class = self.is_class_flag

        self.mmo_base = MultinomialMixtureOnline(self.logs,
                                                 n_clusters,
                                                 is_classification=is_class,
                                                 epsilon=0.01,
                                                 alpha=1.05,
                                                 beta=1.05)
        self.mmo_base.label_logs(self.drain_clusters, self.logs)

    def update_improvement_rate(self, improvement_rate):
        self.improvement_rate = improvement_rate

    def update_is_class_flag(self, is_class_flag):
        self.is_class_flag = is_class_flag

    def get_mmo(self):
        mmo = deepcopy(self.mmo_base)
        mmo.improvement_rate = self.improvement_rate
        return mmo

    def get_em_key(self):
        return 'cem' if self.is_class_flag else 'em'

    def _get_drain_clusters(self, logs, local_drain_params):
        drain = Drain(logs, *local_drain_params)
        drain.parse()
        return drain.cluster_templates

    def _get_n_clusters(self):
        return len(self.drain_clusters) + self.n_clusters_buffer


def run_feedback_convergence(data_configs, drain_parameters, improvement_rates,
                             is_class_flags, constraint_type,
                             n_clusters_buffer=0,
                             n_constraint_samples=5,
                             n_cycles=5,
                             n_samples=1):
    results = {}
    for data_config in data_configs:
        name = data_config['name']
        print('{}...'.format(name))
        results[name] = {}

        run_config = RunConfig(data_config, drain_parameters[name],
                               constraint_type, n_clusters_buffer, n_cycles,
                               n_samples, n_constraint_samples)

        for is_class_flag in is_class_flags:
            run_config.update_is_class_flag(is_class_flag)
            run_config.init_base_mmo()

            em_key = run_config.get_em_key()
            results[name][em_key] = {}

            for ir in improvement_rates:
                print('Rate: {}'.format(ir))
                results[name][em_key][ir] = {}

                run_config.update_improvement_rate(ir)
                acc_vals, t1_vals, t2_vals = perform_runs(run_config)

                results[name][em_key][ir]['acc'] = acc_vals
                results[name][em_key][ir]['t1'] = t1_vals
                results[name][em_key][ir]['t2'] = t2_vals

    return results


def perform_runs(run_config):
    ev = run_config.ev
    logs = run_config.logs
    oracle = run_config.oracle
    n_cycles = run_config.n_cycles
    n_samples = run_config.n_samples
    n_constraint_samples = run_config.n_constraint_samples
    constraint_type = run_config.constraint_type

    acc_vals, t1_vals, t2_vals = [], [], []
    for _ in range(n_samples):
        mmo = run_config.get_mmo()
        acc_vals_sample, t1_vals_sample, t2_vals_sample = [], [], []
        for run_idx in range(n_cycles + 1):
            clusters = mmo.get_clusters(logs)

            acc_val = ev.get_accuracy(clusters)
            t1_val = ev.get_type1_error_ratio()
            t2_val = ev.get_type2_error_ratio()

            acc_vals_sample.append(acc_val)
            t1_vals_sample.append(t1_val)
            t2_vals_sample.append(t2_val)

            if run_idx < n_cycles:
                constraints = oracle.get_constraints(clusters,
                                                     n_constraint_samples, logs)
                mmo.enforce_constraints(constraints, constraint_type)
                mmo.perform_online_batch_em(logs)

        acc_vals.append(acc_vals_sample)
        t1_vals.append(t1_vals_sample)
        t2_vals.append(t2_vals_sample)

    return acc_vals, t1_vals, t2_vals


if __name__ == '__main__':
    improvement_rates = [1.05, 1.50, 2.00]
    constraint_type = None
    is_class_flags = [True, False]
    n_samples = 5

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

    results = run_feedback_convergence(data_configs, drain_parameters,
                                       improvement_rates, is_class_flags,
                                       constraint_type, n_samples=5)
    dump_results('feedback_convergence.p', results)
