import matplotlib.pyplot as plt
from global_utils import load_results
from graphs_final.utils import get_sample_avg, get_sample_std, \
    plot_mean_with_std
from statistics import mean, stdev

results = load_results('exp2_results.p')


def plot_metric(feature, title, ylabel, xlabel):
    varying_sizes = results[feature]['varying_sizes']
    avg_scores_cs = get_sample_avg(results[feature]['ami_cs_samples'])
    std_scores_cs = get_sample_std(results[feature]['ami_cs_samples'])

    base_samples = []
    for samples in results[feature]['ami_base_samples']:
        base_samples.extend(samples)

    mean_base = mean(base_samples)
    std_base = stdev(base_samples)

    mean_base_vector = [mean_base] * len(varying_sizes)
    std_base_vector = [std_base] * len(varying_sizes)

    n_base = len(base_samples)
    n_cs = len(results[feature]['ami_cs_samples'][0])

    plot_mean_with_std(varying_sizes, mean_base_vector, std_base_vector,
                       n_base, 'blue', label='Base')
    plot_mean_with_std(varying_sizes, avg_scores_cs, std_scores_cs,
                       n_cs, 'green', label='Coreset')

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(['Base', 'Coreset'], loc='lower right')
    plt.title(title)
    plt.grid()


plot_metric('cs_ub', 'Performance vs UB Size', 'AMI', 'Coreset UB Size')

plt.subplots_adjust(left=0.1, bottom=0.06, right=0.99,
                    top=0.95, wspace=0.3, hspace=0.5)
plt.show()
