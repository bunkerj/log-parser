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
