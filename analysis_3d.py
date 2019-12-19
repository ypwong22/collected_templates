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

def calc_trend(array, rank):
  array = array.reshape(-1)
  mod = stats.OLS(array, stats.add_constant(range(len(array))))
  reg = mod.fit()
  intercept, slope = reg.params
  p_slope = reg.pvalues[1]
  slope_CI_lower, slope_CI_upper = reg.conf_int()[1, :]
  return (slope, p_slope, slope_CI_lower, slope_CI_upper, rank)

pool = mp.Pool(mp.cpu_count() - 1)
results = [pool.apply_async(calc_trend, args = (var1, ind)) for ind, var1 in enumerate(var)]
pool.close()
pool.join()

trend = np.full(len(data.lat.values) * len(data.lon.values))
for n in range(len(results)):
  trend[retain[results[n][-1]]] = results[n][0]
trend = xr.DataArray(trend.reshape(len(data.lat.values),
                                   len(data.lon.values)),
                     coords = {'lat': data.lat.values, 
                               'lon': data.lon.values},
                     dims = ['lat', 'lon'])
