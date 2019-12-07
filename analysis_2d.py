###############################################################################
# De-trend from each column of a pandas dataframe
###############################################################################
import statsmodels.api as stats

...

data = ... # a pandas data frame

data_detrend = data.copy(deep = True)
for c in range(data.shape[1]):
    x = range(data.shape[0])
    mod = stats.OLS(data.iloc[:, c], stats.add_constant(x))
    results = mod.fit()
    
    data_detrend.iloc[:, c] = data.iloc[:, c] - results.params[1] * x


###############################################################################
# Fit and plot the trend function
###############################################################################
import statsmodels.api as stats
import matplotlib.pyplot as plt


...

time_series = [...]

x = range(len(time_series))
mod = stats.OLS(time_series, stats.add_constant(x))
results = mod.fit()

...

ax.plot(x, results.params[0] + x * results.params[1], ...)
ax.text(0.1, 0.1, ('%f (%.2f)' % (results.params[1], results.pvalues[1])))
