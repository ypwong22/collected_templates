from statsmodels.tsa.stattools import acf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


def panel_trend(ax, vector, pos_xy = [0.1, 0.9],
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
