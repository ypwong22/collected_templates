###############################################################################
# Global mean time series + Climatology diagnostics
###############################################################################
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cartopy.crs as ccrs

def diagnosis(name, path_int, path_out, varname):
    def decode_month_since(time):
        start = time.attrs['units'].split(' ')[2]
        return pd.date_range(start, periods = len(time), freq = '1M')

    hr = xr.open_mfdataset(path_in, decode_times = False)
    var = hr[varname].copy(deep = True)
    if 'month' in hr['time'].attrs['units']:
        var['time'] = decode_month_since(hr['time'])
    else:
        var['time'] = xr.decode_cf(hr)
    hr.close()


    fig = plt.figure(figsize = (6.5, 8))
    gs = gridspec.GridSpec(2, 1, hspace = 0.2, height_ratios = [0.8, 1.2])

    # Time series
    ax = plt.subplot(gs[0])
    ax.plot(var['time'].to_index(), var.mean(dim = ['lat', 'lon']).values)
    ax.set_title(name + ' time series')

    # Map
    ax = plt.subplot(gs[1], projection = ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines()
    cf = ax.contourf(var.lon, var.lat, var.mean(dim = 'time'),
                     cmap = 'Spectral')
    plt.colorbar(cf, ax = ax, orientation = 'horizontal',
                 pad = 0.05)
    ax.set_title(name + ' climatology')

    fig.savefig(path_out, dpi = 600., bbox_inches = 'tight')
    plt.close(fig)


###############################################################################
# Calculate the annual maximum, minimum, and mean for each year and grid point.
###############################################################################
import xarray as xr
import numpy as np

fname = 'myfile.nc'
data = xr.open_dataset(fname, decode_times = True)

def time_mean(array_like):
    """ assume left most dimension is time """
    return np.mean(array_like, axis = 0)
def time_max(array_like):
    """ assume left most dimension is time """
    return np.max(array_like, axis = 0)
def time_min(array_like):
    """ assume left most dimension is time """
    return np.min(array_like, axis = 0)

var_mean = data['var'].resample(indexer = {'time': '1Y'}).apply(time_mean)
var_max = data['var'].resample(indexer = {'time': '1Y'}).apply(time_max)
var_min = data['var'].resample(indexer = {'time': '1Y'}).apply(time_min)

data.close()
