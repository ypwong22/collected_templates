#
import matplotlib.pyplot as plt
import numpy as np

def plot_ts_shade(ax, time, matrix, ts_label = '', ts_col = 'red',
                  shade_col = 'red', alpha = 0.2, skipna = False):
    """
    Plot a shaded ensemble of time series.
    """
    if skipna:
        ts_min = np.nanmin(matrix, axis = 1)
        ts_mean = np.nanmean(matrix, axis = 1)
        ts_max = np.nanmax(matrix, axis = 1)
    else:
        ts_min = np.min(matrix, axis = 1)
        ts_mean = np.mean(matrix, axis = 1)
        ts_max = np.max(matrix, axis = 1)

    hl, = ax.plot(time, ts_mean, '-', color = ts_col, linewidth = 2,
                  label = ts_label)
    ax.plot(time, ts_min, '--', color = shade_col, linewidth = 1)
    ax.plot(time, ts_max, '--', color = shade_col, linewidth = 1)
    ax.fill_between(time, ts_min, ts_max, where = ts_max > ts_min,
                    facecolor = shade_col, alpha = alpha)
    hfill, = ax.fill(np.nan, np.nan, facecolor = shade_col, alpha = alpha)
    return hl, hfill
  
  
# 
import numpy as np
from scipy.stats import norm

def panel_histogram(ax, vector, bins,
                    args_hist = {'color': '#bdbdbd', 'edgecolor': '#636363'}, 
                    args_line = {'color': '#2b8cbe'}):
    """
    Plot a histogram in panel with fitted Gaussian distribution.
    """
    hist, bin_edges = np.histogram(vector, bins = bins, density = True)
    mean, std = norm.fit(hist)
    x = np.linspace(bin_edges[0], bin_edges[-1], 100)
    
    ax.bar(bin_edges[:-1], hist, width = bin_edges[1:] - bin_edges[:-1],
           align = 'edge', **args_hist)
    ax.plot(x, norm.pdf(x, mean, std), **args_line)


###############################################################################
# Colored table according to values
###############################################################################

# Obtain pandas data frame
# df = ...

fig, axes = plt.subplots(figsize=(8,13))
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

norm = plt.Normalize(3, 6)
colours = plt.cm.hot(norm(df.values))
the_table = ax.table(cellText=df.values.astype(int),
                     rowLabels=df.index, colLabels=df.columns,
                     colWidths = [0.1]*vals.shape[1],
                     loc='center', fontsize = 16,
                     cellColours=colours)
