###############################################################################
# Subset a global region using a boolean mask.
###############################################################################
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


# Get mask: True - retain values, False - discard values.
# ! Mask must have same lat & lon dimensions and order as the dataset.
data0 = xr.open_dataset('mask.nc')
mask = data0['mask'].values.copy()
data0.close()

# Get data and apply mask.
data = xr.open_dataset('mydata.nc')
var = xr.DataArray(np.where(mask, data['var'].values.copy(), np.nan), 
                   coords = {'time': data['time'].values,
                             'lat': data['lat'].values,
                             'lon': data['lon'].values},
                   dims = ['time', 'lat', 'lon'])
data.close()

# Save to netcdf.
var.to_dataset(name = 'var').to_netcdf('mydata_subset.nc')

# Accompanying plot.
fig, ax = plt.subplots(subplot_kw = {'projection': ccrs.PlateCarree()})
h = ax.contourf(var.lon, var.lat, var.mean(dim = 'time'))
fig.colorbar(h, ax = ax, cmap = 'Spectral', shrink = 0.7)
fig.savefig('mydata_subset.png')


###############################################################################
# Calculate and plot latitudinal mean time series.
###############################################################################
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

lat_bnd = np.arange(0., 60.1, 0.5)

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
# Calculate and plot mean time series of individual regions defined by levels 
# in a discrete mask.
###############################################################################


