from statistics import mean, mode


def get_sample_statistic(samples, stat):
    return [stat(sample) for sample in samples]


def get_sample_avg(samples):
    return get_sample_statistic(samples, mean)
