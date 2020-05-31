import matplotlib.pyplot as plt

from global_utils import load_results

SUBPLOT_DIM = (4, 4)
LEGEND_IDX = 16
HIGHER_COLOR = 'red'
LOWER_COLOR = 'green'

results = load_results('drain_vs_true_cluster_num.p')


def get_bar_color(dataset_results):
    n_drain = dataset_results['Drain']
    n_truth = dataset_results['Truth']
    if n_drain > n_truth:
        return [HIGHER_COLOR, LOWER_COLOR]
    elif n_drain < n_truth:
        return [LOWER_COLOR, HIGHER_COLOR]
    else:
        return [HIGHER_COLOR, HIGHER_COLOR]


for idx, name in enumerate(results, start=1):
    plt.subplot(*SUBPLOT_DIM, idx)
    plt.title(name)
    plt.grid()
    plt.tight_layout(0.1)
    color = get_bar_color(results[name])
    plt.bar(results[name].keys(), results[name].values(), color=color)

plt.show()
