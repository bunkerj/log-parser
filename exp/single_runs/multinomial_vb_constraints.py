import multiprocessing as mp
from statistics import mean
from exp.utils import get_extended_cs
from src.helpers.oracle import Oracle
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from src.parsers.multinomial_mixture_vb import MultinomialMixtureVB
from global_utils import get_num_true_clusters, get_reduced_assignments
from exp_final.utils import get_log_sample, get_coreset, get_log_strings


def run_multinomial_vb_coreset(data_config, n_constraints, subset_size,
                               proj_dim, n_samples):
    data_manager = DataManager(data_config)
    logs, true_assignments = get_log_sample(data_manager, subset_size)

    n_clusters = get_num_true_clusters(true_assignments)

    w_cs, logs_cs, indices_cs = get_coreset(logs, n_clusters, subset_size,
                                            proj_dim)

    # w_cs, logs_cs, indices_cs = \
    #     get_extended_cs(w_cs, logs_cs, indices_cs)

    true_assignments_cs = get_reduced_assignments(indices_cs,
                                                  true_assignments)
    oracle = Oracle(true_assignments_cs)
    ev = Evaluator(true_assignments)
    # log_strings_cs = get_log_strings(logs, ev.template_truth)

    with mp.Pool(mp.cpu_count()) as pool:
        args = (logs, logs_cs, n_clusters, ev, oracle, n_constraints)
        arg_list = [args for _ in range(n_samples)]
        mp_results = pool.starmap(evaluate_scores, arg_list)
        score_samples, score_cs_samples = list(zip(*mp_results))

    print('Score samples mean: {}'.format(mean(score_samples)))
    print('Score CS samples mean: {}'.format(mean(score_cs_samples)))


def evaluate_scores(logs, logs_cs, n_clusters, ev, oracle, n_constraints):
    mm = MultinomialMixtureVB()
    mm.fit(logs_cs, n_clusters)

    clustering = mm.predict(logs)
    score = ev.get_ami(clustering)

    W = oracle.get_corr_constraints_matrix(
        parsed_clusters=mm.predict(logs_cs),
        n_constraint_samples=n_constraints,
        tokenized_logs=logs_cs,
        weight=1e7)
    mm.fit(logs_cs, n_clusters, p_weights=W, sample_resp=False)

    clustering_const = mm.predict(logs)
    score_const = ev.get_ami(clustering_const)

    return score, score_const


if __name__ == '__main__':
    data_config = DataConfigs.BGL
    n_constraints = 10
    subset_size = 2000
    proj_dim = 2000
    n_samples = 10

    run_multinomial_vb_coreset(data_config, n_constraints, subset_size,
                               proj_dim, n_samples)
