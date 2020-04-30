import matplotlib.pyplot as plt
from global_utils import load_results

results = load_results('drain_accuracy_over_num_logs.p')

plt.plot(results['end_indices'], results['accuracies'])
plt.title('Drain Accuracy Over # Logs')
plt.ylabel('Percentage Accuracy')
plt.xlabel('# Logs')
plt.grid()
plt.show()
