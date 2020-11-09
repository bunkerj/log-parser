import matplotlib.pyplot as plt
from global_utils import load_results
from graphs_final.utils import get_sample_avg, get_sample_std, \
    plot_mean_with_std
from statistics import mean, stdev

METRIC = 'ami'

results = load_results('exp2_results.p')


def plot_metric(feature, title, ylabel, xlabel):
    score_cs_key = '{}_cs_samples'.format(METRIC)
    score_base_key = '{}_base_samples'.format(METRIC)
    varying_sizes = results[feature]['varying_sizes']

    mean_scores_cs = get_sample_avg(results[feature][score_cs_key])
    std_scores_cs = get_sample_std(results[feature][score_cs_key])

    base_samples = []
    for samples in results[feature][score_base_key]:
        base_samples.extend(samples)

    mean_base = mean(base_samples)
    std_base = stdev(base_samples)

    mean_base_vector = [mean_base] * len(varying_sizes)
    std_base_vector = [std_base] * len(varying_sizes)

    n_base = len(base_samples)
    n_cs = len(results[feature][score_cs_key][0])

    plot_mean_with_std(varying_sizes, mean_base_vector, std_base_vector,
                       n_base, 'blue', 'Base')
    plot_mean_with_std(varying_sizes, mean_scores_cs, std_scores_cs,
                       n_cs, 'green', 'Coreset')

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(['Base', 'Coreset'], loc='lower right')
    plt.title(title)
    plt.grid()


plot_metric('cs_ub', 'Performance vs UB Size',
            METRIC.upper(), 'Coreset UB Size')
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99,
                    top=0.95, wspace=0.3, hspace=0.5)
plt.show()
