from global_utils import load_results

PADDING = 20

results = load_results('single_pass_timing.p')

print('{:<15}{}'.format('Name:', results['name']))
print('{:<15}{}'.format('# Clusters:', results['n_clusters']))
print('{:<15}{}'.format('# Vocab:', results['n_vocab']))
print('{:<15}{:.8}'.format('Timing:', results['timing']))
