from random import randint
from skopt import gp_minimize
from exp.utils import update_average_list
from global_utils import dump_results
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from src.parsers.drain import Drain


def run_drain_bayesian_optimization(n_runs, n_calls, acq_func, data_config,
                                    parameter_bounds, name):
    data_manager = DataManager(data_config)
    tokenized_log_entries = data_manager.get_tokenized_log_entries()
    true_assignments = data_manager.get_true_assignments()
    evaluator = Evaluator(true_assignments)

    average_best_accuracy_history = [0] * n_calls

    def get_cumulative_best_accuracies(loss_func_evaluations):
        return [-min(loss_func_evaluations[:i]) for i in
                range(1, len(loss_func_evaluations) + 1)]

    def loss_function(parameters):
        parser = Drain(tokenized_log_entries, *parameters)
        parser.parse()
        return -evaluator.evaluate(parser.cluster_templates)

    for run in range(n_runs):
        res = gp_minimize(loss_function,
                          parameter_bounds,
                          acq_func=acq_func,
                          n_calls=n_calls,
                          n_random_starts=5,
                          random_state=randint(1, 1000000))

        current_best_accuracy_history = \
            get_cumulative_best_accuracies(res.func_vals)

        average_best_accuracy_history = update_average_list(
            average_best_accuracy_history,
            current_best_accuracy_history,
            n_runs)

    dump_results(name, average_best_accuracy_history)


if __name__ == '__main__':
    n_runs = 5
    n_calls = 10
    acq_func = 'EI'
    data_config = DataConfigs.BGL_FULL
    parameter_bounds = [(3, 8), (20, 100), (0.1, 0.6)]
    name = 'drain_bayesian_optimization.p'

    run_drain_bayesian_optimization(n_runs, n_calls, acq_func, data_config,
                                    parameter_bounds, name)
