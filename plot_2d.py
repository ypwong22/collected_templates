import matplotlib.pyplot as plt
import numpy as np

def plot_ts_shade(ax, time, matrix, ts_label = '', ts_col = 'red',
                  shade_col = 'red', alpha = 0.2, skipna = False):
    """
    Plot a shaded ensemble.
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
