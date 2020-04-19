"""
Plot histogram of average timings from feedback_online_evaluation.
"""
import matplotlib.pyplot as plt
from global_utils import load_results

on_results = load_results('feedback_online_em_evaluation_healthapp_10s.p')
off_results = load_results('feedback_offline_em_evaluation_healthapp_10s.p')

on_avg_labeled_timing = on_results['avg_labeled_timing']
on_avg_unlabeled_timing = on_results['avg_unlabeled_timing']
off_avg_labeled_timing = off_results['avg_labeled_timing']
off_avg_unlabeled_timing = off_results['avg_unlabeled_timing']

plt.bar(['Online EM (labeled)',
         'Online EM (unlabeled)',
         'Offline EM (labeled)',
         'Offline EM (unlabeled)'],
        [on_avg_labeled_timing,
         on_avg_unlabeled_timing,
         off_avg_labeled_timing,
         off_avg_unlabeled_timing])
plt.ylabel('Average Timing (Seconds)')
plt.grid()
plt.show()
