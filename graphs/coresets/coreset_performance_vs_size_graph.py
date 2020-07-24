import matplotlib.pyplot as plt
from statistics import mean
from global_utils import load_results

SUBPLOT_DIM = (2, 4)

results = load_results('coreset_performance_vs_size.p')

for idx, name in enumerate(results, start=1):
    plt.subplot(*SUBPLOT_DIM, idx)

    mean_scores_online = []
    mean_scores_coreset = []

    coreset_upper_bound = list(results[name].keys())
    for n in coreset_upper_bound:
        score_samples_online = results[name][n]['score_samples_online']
        score_samples_coreset = results[name][n]['score_samples_coreset']
        mean_scores_online.append(mean(score_samples_online))
        mean_scores_coreset.append(mean(score_samples_coreset))

    plt.title(name)
    plt.plot(coreset_upper_bound, mean_scores_online)
    plt.plot(coreset_upper_bound, mean_scores_coreset)
    plt.grid()

    if idx == (SUBPLOT_DIM[0] * SUBPLOT_DIM[1]):
        plt.legend(['Mean Impurity Online', 'Mean Impurity Coreset'])

plt.show()
