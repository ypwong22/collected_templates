####
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

def ax_trend(ax, vector, pos_xy = [0.1, 0.9],
                args_pt = {'ls': '-'},
                args_ln = {'color': 'k'},
                args_tx = {'color': 'k'}):
    """
    Plot the time series with trend.
    """
    x = np.arange(len(vector))
    temp = ~np.isnan(vector)
    x = x[temp]
    vector = vector[temp]

    ax.plot(x, vector, **args_pt)

    res = OLS(vector, add_constant(x)).fit()
    ax.plot(x, x * res.params[1] + res.params[0], **args_ln)

    ax.text(pos_xy[0], pos_xy[1],
            ppf(res.params[1], res.params[0],
                res.pvalues[1], res.pvalues[0]),
            transform = ax.transAxes, **args_tx)

####
from scipy.stats import gaussian_kde

def ax_scatter_density(ax, x, y, cmap = 'jet'):
    """Plot 2D scatter plot, colored by local density."""
    loc = np.vstack([x.ravel(), y.ravel()])
    f = gaussian_kde(loc)
    density = f(loc)
    h = ax.scatter(x, y, c = density, cmap = cmap,
                   edgecolor = '')
    return h
  
####
from scipy.stats import gaussian_kde

def ax_colored_density(ax, x, y, cmap = 'jet'):
    """Plot 2D Gaussian kde density plot."""
    loc = np.vstack([x.ravel(), y.ravel()])
    f = gaussian_kde(loc)

    x_range = np.linspace(np.min(x) - (np.max(x) - np.min(x))/10,
                          np.max(x) + (np.max(x) - np.min(x))/10, 100)
    y_range = np.linspace(np.min(y) - (np.max(y) - np.min(y))/10,
                          np.max(y) + (np.max(y) - np.min(y))/10, 100)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    z_mesh = f(np.vstack([x_mesh.ravel(), y_mesh.ravel()]))

    cf = ax.contourf(x_range, y_range, 
                     z_mesh.T.reshape(x_mesh.shape), cmap = cmap)
    return cf
