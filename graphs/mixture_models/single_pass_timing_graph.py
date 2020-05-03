from global_utils import load_results

results = load_results('single_pass_timing.p')
print('{}: {}'.format(results['name'], results['timing']))
