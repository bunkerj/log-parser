from exp.mixture_models.utils import get_num_true_clusters
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from src.helpers.oracle import Oracle
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline

N_RUNS = 5
DATA_CONFIG = DataConfigs.HPC

data_manager = DataManager(DATA_CONFIG)
logs = data_manager.get_tokenized_logs()
true_assignments = data_manager.get_true_assignments()
n_true_clusters = get_num_true_clusters(true_assignments)
evaluator = Evaluator(true_assignments)
oracle = Oracle(true_assignments)
parser = MultinomialMixtureOnline(logs,
                                  n_true_clusters,
                                  improvement_rate=1.25,
                                  is_classification=True,
                                  epsilon=0.01,
                                  alpha=1.05,
                                  beta=1.05)

for idx in range(N_RUNS):
    parser.perform_online_batch_em(logs)
    clusters = parser.get_clusters(logs)
    score = evaluator.get_impurity(clusters, [])
    print(score)

    if (idx + 1) < N_RUNS:
        constraints = oracle.get_constraints(clusters, 1, logs)
        parser.enforce_constraints(constraints)
