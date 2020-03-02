"""
Plot the Morris sensitivity indices using a scatter plot.
"""

from utils import load_results
import matplotlib.pyplot as plt


def plot_morris_method_graph(sensitivity_indices, plot_idx, title):
    plt.subplot(1, 2, plot_idx)
    plt.scatter(sensitivity_indices['mu_star'], sensitivity_indices['sigma'],
                marker='x')

    for idx, name in enumerate(sensitivity_indices['names']):
        point = (sensitivity_indices['mu_star'][idx],
                 sensitivity_indices['sigma'][idx])
        plt.annotate(name, point, fontsize=10, ha='center', va='bottom')

    plt.ylabel(r'$\sigma$')
    plt.xlabel(r'$\mu^*$')
    plt.title(title)
    plt.grid()


morris_data = load_results('drain_morris_data.p')
sensitivity_indices = morris_data['accuracy_sens_indices']

plot_morris_method_graph(morris_data['accuracy_sens_indices'], 1, 'Accuracy')
plot_morris_method_graph(morris_data['timing_sens_indices'], 2, 'Timing')

plt.show()
