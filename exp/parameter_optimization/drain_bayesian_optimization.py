from random import randint
from skopt import gp_minimize
from exp.utils import dump_results, update_average_list
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from src.parsers.drain import Drain
from src.utils import get_template_assignments

N_RUNS = 5
N_CALLS = 30
ACQ_FUNC = 'EI'
DATA_CONFIG = DataConfigs.BGL_FULL
PARAMETER_BOUNDS = [(3, 8), (20, 100), (0.1, 0.9)]

data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_log_entries()
true_assignments = get_template_assignments(DATA_CONFIG['assignments_path'])
evaluator = Evaluator(true_assignments)


def get_cumulative_best_accuracies(loss_func_evaluations):
    return [-min(loss_func_evaluations[:i]) for i in
            range(1, len(loss_func_evaluations) + 1)]


def loss_function(parameters):
    parser = Drain(tokenized_log_entries, *parameters)
    parser.parse()
    return -evaluator.evaluate(parser.cluster_templates)


average_best_accuracy_history = [0] * N_CALLS

for run in range(N_RUNS):
    res = gp_minimize(loss_function,
                      PARAMETER_BOUNDS,
                      acq_func=ACQ_FUNC,
                      n_calls=N_CALLS,
                      n_random_starts=5,
                      random_state=randint(1, 1000000))

    current_best_accuracy_history = \
        get_cumulative_best_accuracies(res.func_vals)

    average_best_accuracy_history = update_average_list(
        average_best_accuracy_history,
        current_best_accuracy_history,
        N_RUNS)

filename_template = 'average_best_bayes_opt_accuracy_history_{}.p'
dump_results(filename_template.format(ACQ_FUNC),
             average_best_accuracy_history)
