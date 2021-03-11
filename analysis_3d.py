###############################################################################
# Parallel calculation of trend for each grid of a 3D [time, lat, lon]
# xr.DataAarray.
###############################################################################
import xarray as xr
import numpy as np
import multiprocessing as mp
import statsmodels.api as stats

data = xr.open_dataset('myfile.nc')
var = data['var'].values.copy()
data.close()

var = np.reshape(var, [var.shape[0], var.shape[1] * var_shape[2]])

retain = np.where(~np.isnan(var[0, :]))[0]
var = np.array_split(var[:, retain], len(retain), axis = 1)

def calc_trend(vector, rank):
  vector = vector.reshape(-1)
  mod = stats.OLS(vector, stats.add_constant(range(len(vector))))
  reg = mod.fit()
  intercept, slope = reg.params
  p_slope = reg.pvalues[1]
  slope_CI_lower, slope_CI_upper = reg.conf_int()[1, :]
  return (slope, p_slope, slope_CI_lower, slope_CI_upper, rank)

pool = mp.Pool(mp.cpu_count() - 1)
results = [pool.apply_async(calc_trend, args = (var1, ind)) for ind, var1 in enumerate(var)]
pool.close()
pool.join()

result = [r.get() for r in result]

trend = np.full(len(data.lat.values) * len(data.lon.values))
for n in range(len(results)):
  trend[retain[results[n][-1]]] = results[n][0]
trend = xr.DataArray(trend.reshape(len(data.lat.values),
                                   len(data.lon.values)),
                     coords = {'lat': data.lat.values, 
                               'lon': data.lon.values},
                     dims = ['lat', 'lon'])


###############################################################################
# Easier calculation of trend for each grid of a 3D [time, lat, lon]
# xr.DataAarray.
###############################################################################
import xarray as xr
from scipy.stats import linregress

def one_func(vector):
   result = linregress(np.arange(len(vector)),
                       vector)
   return result.slope

def detrend(sst):
   """ Remove the linear trend from each grid."""
   trend = xr.apply_ufunc(one_func,
                          sst, input_core_dims = [['time']],
                          vectorize = True, dask = 'allowed')
   trend = np.broadcast_to((np.arange(sst.shape[0]) - \
                            (sst.shape[0] - 1)/2).reshape(-1,1,1),
                           sst.shape) * trend.values[np.newaxis, :]
   trend = xr.DataArray(trend, dims = sst.dims,
                        coords = sst.coords)
   #print(trend)
   sst = sst - trend
   return sst, trend

###############################################################################
# Calculate the annual + seasonal average of a xarray DataArray. Assuming the
# time series starts from Jan and ends in Dec. Return DataFrame.
###############################################################################
def seasonal_avg(data_array):
    """
    Calculate the seasonal average of xarray DataArray ('time', 'lat', 'lon')
    """
    result = {}
    result['annual'] = data_array.groupby('time.year').mean(dim = 'time')

    period = temp.to_period(freq = 'Q-NOV')
    for qtr, season in enumerate(['DJF', 'MAM', 'JJA', 'SON'], 1):
        data_temp = data_array[period.quarter == qtr, :, :]
        data_temp['time'] = period[period.quarter == qtr].year
        data_temp = data_temp.groupby('time').mean(dim='time')
        if qtr == 1:
            # Assuming the time series starts from Jan and ends in Dec: 
            # set the seasonal average of 2 months to NaN, and remove the
            # last season (only contains a December).
            data_temp.iloc[0, :, :] = np.nan
            data_temp = data_temp[:-1, :, :]
        result[season] = data_temp
    return result
