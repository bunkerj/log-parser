import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.plots import plot_convergence
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from src.parsers.drain import Drain
from src.utils import get_template_assignments

DATA_CONFIG = DataConfigs.BGL_FULL
PARAMETER_BOUNDS = [(3, 8), (20, 100), (0.1, 0.9)]

data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_log_entries()
true_assignments = get_template_assignments(DATA_CONFIG['assignments_path'])


def loss_function(parameters):
    parser = Drain(tokenized_log_entries, *parameters)
    parser.parse()
    evaluator = Evaluator(true_assignments, parser.cluster_templates)
    return -evaluator.evaluate()


res = gp_minimize(loss_function,
                  PARAMETER_BOUNDS,
                  acq_func='EI',
                  n_calls=30,
                  n_random_starts=5,
                  random_state=123)

plot_convergence(res)
plt.show()
