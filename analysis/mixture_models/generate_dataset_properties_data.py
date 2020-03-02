"""
The goal of this script is to create a labeled dataset to analyze the effect of
providing labeled data to a clustering model.

The predictors are:
1) Vocabulary Size
2) Average Frequency Gini Impurity
3) True Cluster Count
4) Starting Impurity
5) Provided labels Count

The response variable is:
Impurity Percentage Difference
100 * (unlabeled impurity - labeled impurity) / (labeled impurity)
"""
from exp.mixture_models.utils import get_num_true_clusters, \
    get_impurity_difference, normalize_matrix
from global_utils import dump_results, load_results
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from exp.mixture_models.utils import get_avg_gini_impurity
from src.utils import get_template_assignments, get_vocabulary_indices, \
    get_token_counts

data_configs = [
    DataConfigs.Android,
    DataConfigs.Apache,
    DataConfigs.BGL,
    DataConfigs.Hadoop,
    DataConfigs.HDFS,
    DataConfigs.HealthApp,
    DataConfigs.HPC,
    DataConfigs.Linux,
    DataConfigs.Mac,
    DataConfigs.OpenSSH,
    DataConfigs.OpenStack,
    DataConfigs.Proxifier,
    DataConfigs.Spark,
    DataConfigs.Thunderbird,
    DataConfigs.Windows,
    DataConfigs.Zookeeper,
]

data = {}

for data_config in data_configs:
    name = data_config['name']
    data_manager = DataManager(data_config)
    tokenized_log_entries = data_manager.get_tokenized_no_num_log_entries()
    true_assignments = get_template_assignments(data_config['assignments_path'])

    v_indices = get_vocabulary_indices(tokenized_log_entries)
    C = get_token_counts(tokenized_log_entries, v_indices)
    C_probabilities = normalize_matrix(C, 1)
    results = load_results(
        'feedback_evaluation_mp_{}_5s.p'.format(name.lower()))

    labeled_impurities = results['labeled_impurities']
    unlabeled_impurities = results['unlabeled_impurities']
    label_counts = results['label_counts']

    print('{}: {}'.format(name, len(v_indices)))

    data[name] = {}
    data[name]['Vocabulary Size'] = len(v_indices)
    data[name]['True Cluster Count'] = get_num_true_clusters(true_assignments)
    data[name]['Starting Impurity'] = labeled_impurities[0]
    data[name]['Provided labels Count'] = label_counts[2]
    data[name]['Average Frequency Gini Impurity'] = \
        get_avg_gini_impurity(C_probabilities, 1)
    data[name]['Impurity Percentage Difference'] = \
        get_impurity_difference(labeled_impurities[2], unlabeled_impurities[2])

dump_results('dataset_properties_data.p', data)
