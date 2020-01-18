import matplotlib.pyplot as plt

def plot_ts_shade(ax, time, ts = {'min': np.array([]), 'mean': np.array([]),
                                  'max': np.array([])}, ts_label = '',
                  ts_col = 'red'):
    """
    Plot a shaded ensemble.
    """
    hl, = ax.plot(time, ts['mean'], '-', color = ts_col, linewidth = 2,
                  label = ts_label)
    ax.plot(time, ts['min'], '--', color = ts_col, linewidth = 1)
    ax.plot(time, ts['max'], '--', color = ts_col, linewidth = 1)
    ax.fill_between(time, ts['min'], ts['max'],
                    where = ts['max'] > ts['min'],
                    facecolor = ts_col, alpha = 0.2)
    hfill, = ax.fill(np.nan, np.nan, ts_col, alpha = 0.2)
    return hl, hfill
