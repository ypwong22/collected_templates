###############################################################################
# Calculate and plot the global mean time series.
###############################################################################
import xarray as xr
import matplotlib.pyplot as plt

fname = 'myfile.nc'
data = xr.open_dataset(fname, decode_times = True)
time = data['time'].indexes['time']
var_ts = data['var'].mean(dim = ['lat', 'lon'])
data.close()

fig, ax = plt.subplots(figsize = (10, 10))
ax.plot(time, var_ts, '-', color = 'k')
ax.set_xlabel('Year')
ax.set_ylabel('Var [Unit]')
fig.savefig('myfig.png', dpi = 600., bbox_inches = 'tight')
plt.close(fig)


###############################################################################
# Calculate and plot the global climatology.
###############################################################################
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr
import numpy as np


# Calculate the climatology.
fname = 'myfile.nc'
data = xr.open_dataset(fname, decode_times = True)
var = data['var'].mean(dim = ['time'])
data.close()


# Create the figure, get the panel (ax)
...

# Options - Change here
cmap = 'Spectral'
levels = np.linspace(-1., 1., 10)
map_extent = [-180, 180, -60, 90]
grid_on = True # True, False


# Generic module: var - some xr.DataArray
ax.coastlines()
ax.set_extent(map_extent)
h = ax.contourf(var.lon, var.lat, var, cmap = cmap, levels = levels)
plt.colorbar(h, ax = ax, boundaries = levels, ticks = 0.5 * (levels[1:] + levels[:-1]), shrink = 0.7)
if grid_on:
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 20.))
    gl.ylocator = mticker.FixedLocator(np.arange(-90., 91., 10.))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'black', 'weight': 'bold', 'size': 10}
    gl.ylabel_style = {'color': 'black', 'weight': 'bold', 'size': 10}


###############################################################################
# Calculate the annual maximum, minimum, and mean for each year and grid point.
###############################################################################
import xarray as xr
import numpy as np

fname = 'myfile.nc'
data = xr.open_dataset(fname, decode_times = True)

func = np.mean # np.min, np.max, ...
var = data['var'].resample(indexer = {'time': '1Y'}, label = 'right').apply(func)

data.close()

###############################################################################
# Parallel computation of the trend map at each grid.
###############################################################################
import xarray as xr
import numpy as np
import multiprocessing as mp

