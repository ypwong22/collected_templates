import numpy as np

####
from sklearn.neighbors import KernelDensity
from scipy.stats import norm

def ax_histogram(ax, vector, bins, dist = 'norm',
                 args_hist = {'color': '#bdbdbd', 'edgecolor': '#636363'},
                 args_line = {'color': '#2b8cbe'}):
    """
    Plot a histogram in panel with fitted Gaussian distribution.
    dist: *norm* or *kde*
    """
    vector = vector[~np.isnan(vector)]
    hist, bin_edges = np.histogram(vector, bins = bins, density = True)
    ax.bar(bin_edges[:-1], hist, width = bin_edges[1:] - bin_edges[:-1],
           align = 'edge', **args_hist)
    
    x = np.linspace(bin_edges[0], bin_edges[-1], 100)

    if dist == 'norm':
        mean, std = norm.fit(vector)
        ax.plot(x, norm.pdf(x, mean, std), **args_line)
    elif dist == 'kde':
        kde = KernelDensity(bandwidth = np.std(vector)/5, kernel='gaussian')
        kde.fit(vector)
        prob = np.exp(kde.score_samples(x))
        ax.plot(x, prob, '-', **args_line)

####
from statsmodels.tsa.stattools import acf

def ax_acf(ax, vector, max_lag = 12,
              args_bar = {'color': '#bdbdbd', 'edgecolor': '#636363'},
              args_ci = {'color': 'k'}):
    """
    Plot the autocorrelation function of the time series.
    """
    rho, confint, qstat, pvalues = \
        acf(vector[~np.isnan(vector)], unbiased = True, nlags = max_lag, qstat = True,
            alpha = 0.05)

    x = range(1, max_lag + 1)
    ax.bar(x, rho[1:], **args_bar)
    ax.errorbar(x, rho[1:], confint.T[:, 1:] - rho[1:], **args_ci)
