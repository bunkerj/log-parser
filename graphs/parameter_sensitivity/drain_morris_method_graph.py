"""
Plot the Morris sensitivity indices using a scatter plot.
"""
import matplotlib.pyplot as plt
from global_utils import load_results
from graphs.utils import plot_morris_method_graph

morris_data = load_results('drain_morris_data.p')
sensitivity_indices = morris_data['accuracy_sens_indices']

plt.subplot(1, 2, 1)
plot_morris_method_graph(morris_data['accuracy_sens_indices'], 'Accuracy')
plt.subplot(1, 2, 2)
plot_morris_method_graph(morris_data['timing_sens_indices'], 'Timing')

plt.show()
