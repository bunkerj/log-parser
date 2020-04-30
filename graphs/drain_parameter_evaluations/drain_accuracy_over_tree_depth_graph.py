import matplotlib.pyplot as plt
from global_utils import load_results

results = load_results('drain_accuracy_over_tree_depth.p')

plt.plot(results['tree_depths'], results['accuracies'])
plt.title('Drain Accuracy Over Tree Depth')
plt.ylabel('Percentage Accuracy')
plt.xlabel('Tree Depth')
plt.grid()
plt.show()
