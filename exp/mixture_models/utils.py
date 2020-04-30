from random import sample


def get_log_labels(true_assignments, num_of_labels):
    log_labels = {}
    labeled_indices = sample(range(len(true_assignments)), k=num_of_labels)
    for log_idx in labeled_indices:
        cluster = true_assignments[log_idx][-1]
        if cluster not in log_labels:
            log_labels[cluster] = []
        log_labels[cluster].append(log_idx)
    return log_labels


def get_num_true_clusters(true_assignments):
    return len(set(log_data[-1] for log_data in true_assignments))


def split_on_samples(results, n_label_counts):
    """
    Note that the results contain the impurities in the following form:
    [i_0, i_1, ... , i_n, ... , i_0, i_1, ... , i_n], where i_x is a sample
    impurity for the label count with index x. This function splits this single
    list into N lists of the following form [i_0, i_1, ... , i_n], where N is
    the number of sample runs.
    """
    samples = []
    n_samples = len(results) // n_label_counts
    for sample_idx in range(n_samples):
        start_idx = sample_idx * n_label_counts
        end_idx = (sample_idx + 1) * n_label_counts
        samples.append(results[start_idx: end_idx])
    return samples


def split_on_result_sources(results):
    """
    Returns two separate lists: one for labeled results and another for
    unlabeled results.
    """
    lab_impurities = [r[0] for r in results]
    unlab_impurities = [r[1] for r in results]
    return lab_impurities, unlab_impurities


def get_avg(samples):
    n_samples = len(samples)
    sample_len = len(samples[0])
    averages = []
    for label_idx in range(sample_len):
        avg_lab_impurity = 0
        for sample_idx in range(n_samples):
            avg_lab_impurity += samples[sample_idx][label_idx]
        averages.append(avg_lab_impurity / n_samples)
    return averages
