from global_utils import load_results

results = load_results('iplom_exp.p')

results['parser'].print_cluster_templates()

print('Final IPLoM Accuracy: {}'.format(results['accuracy']))
