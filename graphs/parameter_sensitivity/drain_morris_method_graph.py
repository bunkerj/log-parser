"""
Plot the Morris sensitivity indices using a scatter plot.
"""

from graphs.utils import load_results
import matplotlib.pyplot as plt

morris_data = load_results('drain_morris_data.p')
sensitivity_indices = morris_data['accuracy_sens_indices']

plt.scatter(sensitivity_indices['mu_star'], sensitivity_indices['sigma'], marker='x')

for idx, name in enumerate(sensitivity_indices['names']):
    point = (sensitivity_indices['mu_star'][idx],
             sensitivity_indices['sigma'][idx])
    plt.annotate(name, point, fontsize=10, ha='center', va='bottom')

plt.ylabel(r'$\sigma$')
plt.xlabel(r'$\mu^*$')
plt.title('Morris Method Plot')
plt.grid()
plt.show()
