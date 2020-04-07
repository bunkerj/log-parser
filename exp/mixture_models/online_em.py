from exp.mixture_models.utils import get_num_true_clusters
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline

N_ITER = 10
DATA_CONFIG = DataConfigs.Apache

data_manager = DataManager(DATA_CONFIG)
log_entries = data_manager.get_tokenized_no_num_log_entries()
true_assignments = data_manager.get_true_assignments()
num_true_clusters = get_num_true_clusters(true_assignments)

parser = MultinomialMixtureOnline(num_true_clusters, log_entries)

parser.initialize_parameters(log_entries)

for _ in range(N_ITER):
    for log in log_entries:
        parser.process_single_log(log)

template_parsed = parser.get_clusters(log_entries)
evaluator = Evaluator(true_assignments)
accuracy = evaluator.evaluate(template_parsed)

print('Accuracy: {}'.format(accuracy))
