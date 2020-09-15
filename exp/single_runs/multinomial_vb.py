from exp.mixture_models.utils import get_num_true_clusters, get_log_labels
from exp_final.utils import get_log_sample
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from src.helpers.oracle import Oracle
from src.parsers.multinomial_mixture_vb import MultinomialMixtureVB


def run_multinomial_vb(data_config, subset_size, n_labels):
    data_manager = DataManager(data_config)
    logs, true_assignments = get_log_sample(data_manager, subset_size)

    n_clusters = get_num_true_clusters(true_assignments)
    ev = Evaluator(true_assignments)
    log_labels = get_log_labels(true_assignments, n_labels)
    oracle = Oracle(true_assignments)

    mm = MultinomialMixtureVB()
    mm.fit(logs, n_clusters)
    clustering = mm.predict(logs)
    print(ev.get_ami(clustering))

    W = oracle.get_constraints_matrix(
        parsed_clusters=clustering,
        n_constraint_samples=n_labels,
        tokenized_logs=logs,
        weight=1)
    mm.fit(logs, n_clusters, log_labels=log_labels, p_weights=W)
    clustering = mm.predict(logs)
    print(ev.get_ami(clustering))


if __name__ == '__main__':
    data_config = DataConfigs.BGL_FULL_FINAL
    subset_size = 50000
    n_labels = 5000

    run_multinomial_vb(data_config, subset_size, n_labels)
