import matplotlib.pyplot as plt

from global_utils import load_results

results = load_results('exp3_results.p')

N = len(results['label_counts'])
drain_scores = N * [results['drain_score']]

plt.subplot(1, 2, 1)
plt.plot(results['label_counts'], results['label_score_samples'])
plt.plot(results['label_counts'], drain_scores)
plt.legend(['Multinomial VB', 'Drain'])
plt.ylabel('AMI')
plt.xlabel('Label Counts')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(results['constraint_counts'], results['constraint_score_samples'])
plt.plot(results['label_counts'], drain_scores)
plt.ylabel('AMI')
plt.xlabel('Constraint Counts')
plt.grid()

plt.show()
