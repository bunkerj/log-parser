from global_utils import get_log_labels, get_num_true_clusters
from exp_final.utils import get_log_sample
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from src.helpers.oracle import Oracle
from src.helpers.cluster_tracker import ClusterTracker
from src.parsers.multinomial_mixture_vb import MultinomialMixtureVB


def run_cluster_trajectories(data_config, subset_size, n_labels, n_constraints):
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

    W = oracle.get_corr_constraints_matrix(
        parsed_clusters=clustering,
        n_constraint_samples=n_constraints,
        tokenized_logs=logs,
        weight=1000)
    mm.init(logs, n_clusters, log_labels=log_labels, p_weights=W)

    tracker = ClusterTracker(logs, mm, n_clusters, list(range(25)))
    tracker.run(10)
    tracker.show()

    clustering = mm.predict(logs)
    print(ev.get_ami(clustering))


if __name__ == '__main__':
    data_config = DataConfigs.BGL
    subset_size = 2000
    n_labels = 0
    n_constraints = 500

    run_cluster_trajectories(data_config, subset_size, n_labels, n_constraints)
