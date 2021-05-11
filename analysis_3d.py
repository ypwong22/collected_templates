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
    """ mask: [lon_min, lon_max, lat_min, lat_max] """
    lon_mask = (da['lon'].values >= mask[0]) & (da['lon'].values <= mask[1])
    lat_mask = (da['lat'].values >= mask[2]) & (da['lat'].values <= mask[3])
    lon_mask, lat_mask = np.meshgrid(lon_mask, lat_mask)
    return da.where(lon_mask & lat_mask)
