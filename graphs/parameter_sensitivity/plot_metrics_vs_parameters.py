"""
Plot the various parameters against both their corresponding accuracies and
timings.
"""
from utils import load_results
import matplotlib.pyplot as plt

morris_data = load_results('drain_morris_data.p')


def plot_graphs_for_single_metric(data, metric_name, row_idx, n_rows):
    for parameter_idx, name in enumerate(data['parameter_names']):
        n_col = data['parameters'].shape[1]
        subplot_idx = (row_idx - 1) * n_col + parameter_idx + 1
        plt.subplot(n_rows, n_col, subplot_idx)
        plt.title('{} vs {}'.format(metric_name.capitalize(), name))
        plt.xlabel(name)
        plt.ylabel(metric_name.capitalize())
        plt.grid()
        for idx, accuracy in enumerate(data[metric_name]):
            parameter = data['parameters'][idx][parameter_idx]
            plt.scatter(parameter, accuracy, marker='x')


plt.title('Scatter Plots of Timings and Accuracy over Parameters')
plot_graphs_for_single_metric(morris_data, 'accuracy', 1, 2)
plot_graphs_for_single_metric(morris_data, 'timing', 2, 2)
plt.tight_layout()
plt.show()
