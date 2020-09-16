import matplotlib.pyplot as plt
from global_utils import load_results
from graphs_final.utils import get_sample_avg

results = load_results('exp4_results.p')

N = len(results['cs_size'])
avg_rand_label_scores = get_sample_avg(results['rand_label_ami_samples'])
full_avg_rand_label_score = N * get_sample_avg([avg_rand_label_scores])
avg_cs_label_scores = get_sample_avg(results['cs_label_ami_samples'])

plt.plot(results['cs_size'], full_avg_rand_label_score)
plt.plot(results['cs_size'], avg_cs_label_scores)
plt.legend(['Random', 'Coreset'])
plt.xlabel('Coreset Size')
plt.ylabel('AMI')
plt.title('Performance vs Coreset Size')
plt.grid()
plt.plot()

plt.show()
