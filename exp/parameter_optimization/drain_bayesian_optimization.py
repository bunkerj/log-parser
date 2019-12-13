from skopt import gp_minimize
from exp.utils import dump_results
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from src.parsers.drain import Drain
from src.utils import get_template_assignments

N_RUNS = 30
DATA_CONFIG = DataConfigs.BGL_FULL
PARAMETER_BOUNDS = [(3, 8), (20, 100), (0.1, 0.9)]

data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_log_entries()
true_assignments = get_template_assignments(DATA_CONFIG['assignments_path'])


def get_cumulative_best_accuracies(loss_func_evaluations):
    return [-min(loss_func_evaluations[:i]) for i in
            range(1, len(loss_func_evaluations) + 1)]


def loss_function(parameters):
    parser = Drain(tokenized_log_entries, *parameters)
    parser.parse()
    evaluator = Evaluator(true_assignments, parser.cluster_templates)
    return -evaluator.evaluate()


res = gp_minimize(loss_function,
                  PARAMETER_BOUNDS,
                  acq_func='EI',
                  n_calls=N_RUNS,
                  n_random_starts=5,
                  random_state=123)

best_accuracy_history = get_cumulative_best_accuracies(res.func_vals)

dump_results('best_bayes_opt_accuracy_history.p', best_accuracy_history)
