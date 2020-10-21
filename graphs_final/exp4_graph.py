import numpy as np
import matplotlib.pyplot as plt
from global_utils import load_results
from graphs_final.utils import get_sample_avg, get_sample_std

results = load_results('exp4_results.p')

N = len(results['cs_size'])
cs_sizes_arr = np.array(results['cs_size'])

avg_rand_label_scores = get_sample_avg(results['rand_label_ami_samples'])
avg_cs_label_scores = get_sample_avg(results['cs_label_ami_samples'])
std_rand_label_scores = get_sample_std(results['rand_label_ami_samples'])
std_cs_label_scores = get_sample_std(results['cs_label_ami_samples'])

m_rand, b_rand = np.polyfit(cs_sizes_arr, avg_rand_label_scores, 1)
m_cs, b_cs = np.polyfit(cs_sizes_arr, avg_cs_label_scores, 1)

plt.errorbar(cs_sizes_arr, avg_rand_label_scores, std_rand_label_scores,
             linestyle='None', marker='^')
plt.errorbar(cs_sizes_arr, avg_cs_label_scores, std_cs_label_scores,
             linestyle='None', marker='^')

plt.plot(cs_sizes_arr, m_rand * cs_sizes_arr + b_rand)
plt.plot(cs_sizes_arr, m_cs * cs_sizes_arr + b_cs)

plt.legend(['Random', 'Coreset'])
plt.xlabel('Coreset Size')
plt.ylabel('AMI')
plt.title('Performance vs Coreset Size')
plt.grid()
plt.plot()

plt.show()
