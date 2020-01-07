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


###############################################################################
# Calculate the annual + seasonal average of a time series. Assuming the
# time series starts from Jan and ends in Dec.
###############################################################################
import pandas as pd
import numpy as np

def seasonal_avg(pd_series):
    result = {}
    result['annual'] = pd_series.groupby(pd_series.index.year).mean()

    temp = pd_series.index.to_period(freq = 'Q-NOV')
    temp2 = pd_series.groupby(temp).mean()

    # Assuming the time series starts from Jan and ends in Dec: 
    # set the seasonal average of 2 months to NaN, and remove the
    # last season (only contains a December)
    temp2.iloc[0] = np.nan
    temp2 = temp2.iloc[:-1]

    for qtr, season in enumerate(['DJF', 'MAM', 'JJA', 'SON'], 1):
        result[season] = temp2.loc[temp2.index.quarter == qtr]
        result[season].index = result[season].index.year
        
    return pd.DataFrame(result)
