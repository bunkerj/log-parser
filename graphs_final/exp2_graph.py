import matplotlib.pyplot as plt
from global_utils import load_results
from graphs_final.utils import get_sample_avg
from statistics import mean

results = load_results('exp2_results.p')


def plot_metric(subplot_args, feature, title, ylabel, xlabel):
    varying_sizes = results[feature]['varying_sizes']
    N = len(varying_sizes)
    avg_scores_cs = get_sample_avg(results[feature]['ami_cs_samples'])
    avg_scores_base = get_sample_avg(results[feature]['ami_base_samples'])
    const_avg_score_base = N * [mean(avg_scores_base)]

    plt.subplot(*subplot_args)
    plt.plot(varying_sizes, const_avg_score_base)
    plt.plot(varying_sizes, avg_scores_cs)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if subplot_args[2] == 1:
        plt.legend(['Base', 'Coreset'])
    plt.title(title)
    plt.grid()


plot_metric((1, 2, 1), 'cs_ub', 'Performance vs UB Size',
            'AMI', 'Coreset UB Size')
plot_metric((1, 2, 2), 'cs_proj', 'Performance vs Proj Size',
            'AMI', 'Coreset Proj Size')
plt.show()
