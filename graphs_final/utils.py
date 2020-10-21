from statistics import mean, stdev


def get_sample_statistic(samples, stat):
    return [stat(sample) for sample in samples]


def get_sample_avg(samples):
    return get_sample_statistic(samples, mean)


def get_sample_std(samples):
    return get_sample_statistic(samples, stdev)
