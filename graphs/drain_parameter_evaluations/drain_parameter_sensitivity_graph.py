import matplotlib.pyplot as plt
from global_utils import load_results

boxplot_data = load_results('drain_parameter_sensitivity.p')

plt.boxplot(boxplot_data, sym='')
plt.title('Drain Parameter Sensitivity')
plt.ylabel('Percentage Accuracy')
plt.xticks(range(1, 4), ['max_depth', 'max_child', 'sim_threshold'])
plt.grid()
plt.show()
