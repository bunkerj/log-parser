import matplotlib.pyplot as plt
from global_utils import load_results
from graphs_final.utils import get_sample_avg

results = load_results('exp3_results.p')

N = len(results['label_counts'])
drain_scores = N * [results['drain_score']]

plt.subplot(1, 2, 1)
avg_label_scores = get_sample_avg(results['label_ami_samples'])
plt.plot(results['label_counts'], avg_label_scores)
plt.plot(results['label_counts'], drain_scores)
plt.legend(['Multinomial VB', 'Drain'])
plt.ylabel('AMI')
plt.xlabel('Label Counts')
plt.title('Performance vs Label Count')
plt.grid()

plt.subplot(1, 2, 2)
avg_constraint_scores = get_sample_avg(results['constraint_ami_samples'])
plt.plot(results['constraint_counts'], avg_constraint_scores)
plt.plot(results['label_counts'], drain_scores)
plt.ylabel('AMI')
plt.xlabel('Constraint Counts')
plt.title('Performance vs Constraint Count')
plt.grid()

plt.show()
