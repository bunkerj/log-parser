import matplotlib.pyplot as plt
from global_utils import load_results

results = load_results('drain_accuracy_over_similarity.p')

plt.plot(results['sim_thresholds'], results['accuracies'])
plt.plot(results['sim_thresholds'], results['type1_error_ratios'])
plt.plot(results['sim_thresholds'], results['type2_error_ratios'])
plt.legend(['Total Accuracy', 'Type 1 Error Ratio', 'Type 2 Error Ratio'])
plt.title('Error over Similarity Threshold')
plt.ylabel('Percentage')
plt.xlabel('Similarity Threshold')
plt.grid()
plt.show()
