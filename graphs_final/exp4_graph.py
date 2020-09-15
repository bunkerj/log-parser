import matplotlib.pyplot as plt
from global_utils import load_results
from graphs_final.utils import get_sample_avg

results = load_results('exp4_results.p')

N = len(results['cs_size'])
avg_rand_label_scores = get_sample_avg(results['random_label_scores'])
full_avg_rand_label_score = N * get_sample_avg([avg_rand_label_scores])
avg_cs_label_scores = get_sample_avg(results['coreset_label_scores'])

plt.plot(results['cs_size'], full_avg_rand_label_score)
plt.plot(results['cs_size'], avg_cs_label_scores)
plt.legend(['Random', 'Coreset'])
plt.xlabel('Coreset Size')
plt.ylabel('AMI')
plt.grid()
plt.plot()

plt.show()
