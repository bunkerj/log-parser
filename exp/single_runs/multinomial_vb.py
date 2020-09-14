from exp.mixture_models.utils import get_num_true_clusters, get_log_labels
from exp_final.utils import get_log_sample
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from src.parsers.multinomial_mixture_vb import MultinomialMixtureVB


def run_multinomial_vb(data_config, subset_size, n_labels):
    data_manager = DataManager(data_config)
    logs, true_assignments = get_log_sample(data_manager, subset_size)

    n_clusters = get_num_true_clusters(true_assignments)
    ev = Evaluator(true_assignments)
    log_labels = get_log_labels(true_assignments, n_labels)

    mm = MultinomialMixtureVB()
    mm.fit(logs, n_clusters, log_labels=log_labels)
    clustering = mm.predict(logs)

    return ev.get_nmi(clustering)


if __name__ == '__main__':
    data_config = DataConfigs.BGL_FULL_FINAL
    subset_size = 50000
    n_labels = 5000

    results = run_multinomial_vb(data_config, subset_size, n_labels)
    print(results)
