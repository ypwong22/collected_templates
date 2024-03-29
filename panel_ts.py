import numpy as np
import pandas as pd


####
def ax_shade(ax, time, matrix, ts_label = '', ts_col = 'red',
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


####
def ax_oscillation(ax, inx_series):
    """A script to generate the common blue-red style ocean indices plot"""
    ax.fill_between(inx_series.index, inx_series.values, 0,
                    where = inx_series.values >= 0, color = 'r',
                    interpolate = True)
    ax.fill_between(inx_series.index, inx_series.values, 0,
                    where = inx_series.values <= 0, color = 'b',
                    interpolate = True)
    ax.plot(inx_series.values, inx_series.values, '-k', lw = 0.5)


