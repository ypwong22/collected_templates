import numpy as np

####
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.stats import pearsonr

from tools.format_text import ppf, ppp

def ax_regress(ax, x, vector, 
               display = 'equation',
               pos_xy = [0.1, 0.9],
               args_pt = {'ls': '-'},
               args_ln = {'color': 'k'},
               args_ci = {'color': 'k', 'alpha': 'shade'},
               args_tx = {'color': 'k'}):
    """
    Plot the time series with trend.

    Parameters
    ----------
    ax: matplotlib.pyplot.axis
    x: 1-d array
        The x-values in the regression.
    vector: 1-d array
        The y-values in the regression.
    display: None or str
        If None, does not display the regression equation.
        If 'equation', display the regression equation.
        If 'pearson', display the Pearson correlation.
    pos_xy: [float, float]
        The position to place the annotation in the normalized axis unit.
    args_pt, args_ln, args_ci, args_txt: dict
        Keyword arguments to be passed into the scatter plot, regression
        line, confidence interval for the regression line, and annotation
        text plotting functions.
    """
    temp = (~np.isnan(vector)) & (~np.isnan(x))
    x = x[temp]
    vector = vector[temp]

    ax.plot(x, vector, **args_pt)

    reg = OLS(vector, add_constant(x)).fit()
    ax.plot(x, x * reg.params[1] + reg.params[0], **args_ln)

    _, predict_ci_low, predict_ci_upp = wls_prediction_std(reg, \
        exog = reg.model.exog, weights = np.ones(len(reg.model.exog)))
    x_ind = np.argsort(x)
    ax.fill_between(x[x_ind], predict_ci_low[x_ind], 
                    predict_ci_upp[x_ind], interpolate = True,
                    **args_ci)

    if display == 'equation':
        ax.text(pos_xy[0], pos_xy[1],
                ppf(reg.params[1], reg.params[0],
                    reg.pvalues[1], reg.pvalues[0]),
                transform = ax.transAxes, **args_tx)
    elif display == 'pearson':
        r, pval = pearsonr(x, temp)
        ax.text(pos_xy[0], pos_xy[1],
                ('%.3f' % r) + ppp(pval),
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
