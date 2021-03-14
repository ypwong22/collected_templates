import numpy as np

####
from sklearn.neighbors import KernelDensity
from scipy.stats import norm

def ax_histogram(ax, vector, bins, dist = 'norm',
                 show = 'both',
                 args_hist = {'color': '#bdbdbd', 'edgecolor': '#636363'},
                 args_line = {'color': '#2b8cbe'}):
    """
    Plot a histogram in panel with fitted Gaussian distribution.

    Parameters
    ----------
      vector: 1-d array
          The data set.
      bins: int or a sequence of increasing numbers
          The same as input into **numpy.histogram**.
          Defines the number of bins or the bin edges, from left-most to
          right-most.
      dist: str
          If *norm*, plot fitted normal distribution.
          If *kde*, plot fitted Gaussian kernal density.
      show: str
          If *both* (default), plot both the histogram bars and fitted
          distribution.
          If *bar*, plot only the histogram bars.
          If *line*, plot only the fitted distribution using lines.
    """
    vector = vector[~np.isnan(vector)]
    hist, bin_edges = np.histogram(vector, bins = bins, density = True)

    h = []

    if (show == 'bar') | (show == 'both'):
        h1 = ax.bar(bin_edges[:-1], hist,
                    width = bin_edges[1:] - bin_edges[:-1],
                    align = 'edge', **args_hist)
        h.append(h1)

    if (show == 'line') | (show == 'both'):
        x = np.linspace(bin_edges[0], bin_edges[-1], 100)

        if dist == 'norm':
            mean, std = norm.fit(vector)
            h2, = ax.plot(x, norm.pdf(x, mean, std), **args_line)
        elif dist == 'kde':
            kde = KernelDensity(bandwidth = np.std(vector)/5,
                                kernel='gaussian')
            kde.fit(vector)
            prob = np.exp(kde.score_samples(x))
            h2, = ax.plot(x, prob, '-', **args_line)
        h.append(h2)

    return h


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
