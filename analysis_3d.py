###############################################################################
# Fast calculation of trend for each grid of a 3D [time, lat, lon]
# xr.DataAarray.
###############################################################################
import xarray as xr
from scipy.stats import linregress
from utils.tools.unit_funcs import unit_trend

def detrend(sst):
   """ Remove the linear trend from each grid."""
   trend = xr.apply_ufunc(unit_trend,
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
# Fast calculation of trend for each grid of a 3D [time, lat, lon]
# np.ma.array.
###############################################################################
import warnings

def _normalize(ma_array):
    temp = np.where(ma_array.mask, np.nan, ma_array.data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)
        n_mean = np.nanmean(temp, axis = 0, keepdims = True)
        n_std = np.nanstd(temp, axis = 0, keepdims = True)

    temp = (temp - n_mean) / np.where((n_std > 0.) | np.isnan(n_std), n_std, 
                                       np.nanmin(n_std[n_std > 0.]) * 1e-3)
    temp = np.ma.masked_where(ma_array.mask, temp)
    return temp, n_mean, n_std

def _olsTensor(Y, x):
    """ Repeated calculation of linear regression in the spatial dimensions.
    Parameters
    ----------
    Y : np.ma.array
        The variable of interest. The first dimension will be assumed to be
        time (replicate observations).
    x : np.array or np.ma.array
        The time variable of interest. If one-dimensional, will be propagated
        to the dimensionality of Y. If having the same dimensionality as Y,
        must be a masked array.
    Returns
    -------
    r : np.ma.array
        The trend. If x only has a time dimension, `r` is a scalar.
        Otherwise, `r` has the same dimensionality as x[1:].
    p : np.ma.array
        The two-sided p-values of the trend. If x only has a time 
        dimension, `p` is a scalar. Otherwise, `p` has the same 
        dimensionality as x[1:].
    """
    if type(Y) != np.ma.core.MaskedArray:
        raise TypeError('Y must be a masked array')
    if Y.shape[0] < 3:
        raise ValueError('At least three observations are needed')

    if (type(x) != np.ma.core.MaskedArray) and (type(x) != np.ndarray):
        raise TypeError('x must be either masked or ordinary numpy array')
    if (not np.allclose(x.shape, Y.shape)) and (len(x.shape) != 1):
        raise ValueError('x must be either 1-dimensional or has the same shape as Y')

    # homogenize the shape and mask of x and Y
    if type(Y.mask) == bool:
        Y.mask = np.full(Y.shape, Y.mask)
    if type(x) == np.ma.core.MaskedArray:
        if type(x.mask) == bool:
            x.mask = np.full(x.shape, x.mask)
    else:
        x = np.ma.array(x, mask = np.full(x.shape, False))

    orig_shape = Y.shape
    Y = Y.reshape(Y.shape[0], 1, int(np.prod(Y.shape[1:])))
    if len(x.shape) != 1:
        x = x.reshape(Y.shape)
    else:
        x = np.ma.array(np.broadcast_to(x.data.reshape(-1,1,1), Y.shape),
                        mask = np.broadcast_to(x.mask.reshape(-1,1,1), Y.shape))
    x = np.ma.array(x.data, mask = x.mask | Y.mask)
    Y = np.ma.array(Y, mask = x.mask)

    # normalize
    x, _, x_scale = _normalize(x)
    Y, _, Y_scale = _normalize(Y)

    # add constant term
    x = np.ma.concatenate([np.ma.array(np.ones(Y.shape), mask = Y.mask), x], axis = 1)

    # calculate the regression coefficients; treating the masked points as if zero.
    xx = np.where(x.mask == False, x.data, 0.)
    yy = np.where(Y.mask == False, Y.data, 0.)
    beta = np.einsum('ijk,jlk->ilk',
                     np.einsum('ijk,ljk->ilk',
                               np.linalg.pinv(np.einsum('ijk,ilk->jlk',xx,xx \
                                                    ).transpose(2,0,1)).transpose(1,2,0),
                               xx), yy)
    # calculate the p-value
    from scipy.stats import t
    dof = np.sum(np.ma.getmaskarray(Y)== False, axis = 0) - 2
    resid = yy - np.einsum('ijk,jlk->ilk', xx, beta)
    mse = np.sum(np.power(resid,2), axis=0) / dof
    # somehow, unable to apply np.ma.mean on x[:,[1],:]
    temp = x[:,[1],:]
    temp.data[temp.mask] = np.nan
    temp = temp.data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)
        std = np.nansum(np.power(temp - np.nanmean(temp, axis = 0, keepdims = True), 2), axis = 0)

    # somehow, using masked array here results in underflow error; had to use np.nan
    np.seterr(divide='ignore', invalid='ignore')
    beta = beta[1, :] # discard intercept
    tval = beta / np.sqrt(mse/std)
    np.seterr(divide='raise', invalid='raise')
    pval = 2 * t.sf(np.abs(tval), dof)

    # scale the beta
    beta = beta * Y_scale / x_scale

    # mask the data
    tval = np.ma.masked_invalid(tval)
    pval = np.ma.array(pval, mask = tval.mask)
    beta = np.ma.array(beta, mask = tval.mask)

    # restore shape
    if len(orig_shape) > 1:
        beta = beta.reshape(orig_shape[1:])
        pval = pval.reshape(orig_shape[1:])
    else:
        beta = float(beta.data)
        pval = float(pval.data)
    return beta, pval


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


##################################################################################
# Calculate the grid-level climatology, seasonality, trend, and anomalies
# for an xarray.DataArray, whose dimensions are ['time', 'lat', 'lon']
##################################################################################
import numpy as np
import xarray as xr
# use olsTensor

def _grid_decompose(da):
   da.load() # makes things faster

   clim          = da.mean(dim = 'time')
   seasonal_anom = da.groupby('time.month').mean()
   max_month     = seasonal_anom.idxmax(dim = 'month')

   anomalies     = da.copy(deep = True)
   for i in range(1, 13):
      anomalies[(i-1)::12, :, :] = anomalies[(i-1)::12, :, :].values - \
          clim.values - seasonal_anom[i-1, :, :].values

   temp2         = np.ma.MaskedArray(anomalies.values, np.isnan(anomalies.values)).reshape(-1, 12, anomalies.shape[1], anomalies.shape[2])
   x             = np.arange(temp2.shape[0]) - np.mean(np.arange(temp2.shape[0]))
   beta, pval    = olsTensor(temp2, x)

   trend         = xr.DataArray(beta, dims = ['month', 'lat', 'lon'],
                              coords = {'month': seasonal_anom['month'],
                                      'lat'  : seasonal_anom['lat'], 
                                      'lon'  : seasonal_anom['lon']})

   for i in range(1, 13):
      anomalies[(i-1)::12, :, :] = anomalies[(i-1)::12, :, :].values - x.reshape(-1, 1, 1) * beta[[i-1], :, :]
   anomalies_std = anomalies.std(dim = 'time')

   return clim, max_month, trend, anomalies_std


##################################################################################
# Calculate the climatology, seasonality, trend, and anomalies on the 
# averaged time series of an xarray.DataArray, whose dimensions are 
# ['time', 'lat', 'lon']
##################################################################################
import numpy as np
import pandas as pd
# use olsTensor

def _ts_decompose(self, da):
   weights       = np.cos(da['lat'] * np.pi / 180)
   da_series     = da.weighted(weights).mean(['lat', 'lon'])

   clim          = float(da_series.mean())
   annual_avg    = da_series.resample({'time': '1Y'}).mean()

   seasonal_anom = da_series.groupby('time.month').mean() - clim
   seasonal_anom = pd.Series(seasonal_anom.values, index = seasonal_anom['month'].values)

   temp          = da_series.values.reshape(-1, 12) - clim - seasonal_anom.values.reshape(1, 12)

   temp2         = np.ma.MaskedArray(temp, False)
   x             = np.arange(temp2.shape[0]) - np.mean(np.arange(temp2.shape[0]))
   beta, pval    = olsTensor(temp2, x)
   trend         = pd.Series(beta, index = range(1,13))

   anomalies     = temp - x.reshape(-1,1) * trend.reshape(1, 12)
   anomalies     = pd.DataFrame(anomalies, index = np.unique(da_series['time'].to_index().year),
                               columns = range(1, 13))
   anomalies_std = anomalies.std(axis = 1)

   return annual_avg, seasonal_anom, trend, anomalies_std


###############################################################################
# Frequency, Intensity, Mean of daily to monthly precipitation.
###############################################################################
frequency = (da > 0.).resample({'time': '1M'}).mean()
intensity = (da.where(da > 0.)).resample({'time': '1M'}).mean()
mean = da.resample({'time': '1M'}).mean()


###############################################################################
# Interpolate data array (if doesn't care about missing boundary grids).
###############################################################################
target_lat = np.arange(-89.75, 89.76, 0.5)
target_lon = np.arange(-179.75, 179.76, 0.5)
da.interp(coords = {'lon': target_lon, 'lat': target_lat}).load()


###############################################################################
# Flip longitude from 360 to -180 to 180.
###############################################################################
def flip_lon(da):
    lon_dim = [dd for dd in da.dims if dd.lower() in ['lon','longitude']][0]
    da = da.roll({lon_dim: sum(da[lon_dim].values > 180)},
                 roll_coords = True)
    da[lon_dim] = np.where(da[lon_dim] > 180,
                           da[lon_dim] - 360, da[lon_dim])
    return da

###############################################################################
# Flip longitude from -180 to 180 to 360.
###############################################################################
def flip_lon2(da):
    lon_dim = [dd for dd in da.dims if dd.lower() in ['lon','longitude']][0]
    da = da.roll({lon_dim: - sum(da[lon_dim].values < 0)},
                 roll_coords = True)
    da[lon_dim] = np.where(da[lon_dim] < 0,
                           da[lon_dim] + 360, da[lon_dim])
    return da

###############################################################################
# Apply the rectangular mask on the data array.
###############################################################################
def apply_rect_mask(da, mask):
    """ mask: [lon_min, lat_min, lon_max, lat_max] """
    lon_mask = (da['lon'].values >= mask[0]) & (da['lon'].values <= mask[2])
    lat_mask = (da['lat'].values >= mask[1]) & (da['lat'].values <= mask[3])
    lon_mask, lat_mask = np.meshgrid(lon_mask, lat_mask)
    return da.where(lon_mask & lat_mask)
