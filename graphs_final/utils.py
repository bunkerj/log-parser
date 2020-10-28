import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev


def get_sample_statistic(samples, stat):
    return [stat(sample) for sample in samples]


def get_sample_avg(samples):
    return get_sample_statistic(samples, mean)


def get_sample_std(samples):
    return get_sample_statistic(samples, stdev)


def plot_mean_with_std(x_axis, mu, s, n, color, label):
    mu_arr = np.array(mu)
    s_arr = np.array(s)
    plt.plot(x_axis, mu_arr, lw=2, label=label, color=color)
    s_norm = s_arr / np.sqrt(n)
    ub = mu_arr + 2 * s_norm
    lb = mu_arr - 2 * s_norm
    plt.fill_between(x_axis, lb, ub, facecolor=color, alpha=0.5)
