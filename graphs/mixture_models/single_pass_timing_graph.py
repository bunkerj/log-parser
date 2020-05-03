from global_utils import load_results

PADDING = 20

results = load_results('single_pass_timing.p')

print('Name:'.ljust(PADDING) + '{}'.format(results['name']))
print('Clusters:'.ljust(PADDING) + '{}'.format(results['n_clusters']))
print('Timing:'.ljust(PADDING) + '{:.8}'.format(results['timing']))
