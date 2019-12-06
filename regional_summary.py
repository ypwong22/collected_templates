###############################################################################
# Subset a global region using a boolean mask.
###############################################################################
import xarray as xr
import numpy as np
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

# Cut down the dataset to only a bounding box of the mask.
eps = 0.01

mask_lat = data.lat.values[np.isfinite(mask).any(axis = 1)]
mask_lon = data.lon.values[np.isfinite(mask).any(axis = 0)]
var = var.sel(lat = slice(np.min(mask_lat) - eps, np.max(mask_lat) + eps),
              lon = slice(np.min(mask_lon) - eps, np.max(mask_lon) + eps))

# Save to netcdf.
var.to_dataset(name = 'var').to_netcdf('mydata_subset.nc')

# Accompanying plot.
fig, ax = plt.subplots(subplot_kw = {'projection': ccrs.PlateCarree()})
ax.coastlines()
h = ax.contourf(var.lon, var.lat, var.mean(dim = 'time'))
fig.colorbar(h, ax = ax, shrink = 0.7)
fig.savefig('mydata_subset.png')
plt.close(fig)

###############################################################################
# Calculate and plot latitudinal mean time series.
###############################################################################
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

lat_bnd = np.arange(0., 60.1, 0.5)
eps = 0.01 # Tolerance for floating point latitude comparison.

var_bylat = np.full(len(lat_bnd)-1, np.nan)

data = xr.open_dataset('myfile.nc')
for i in range(len(lat_bnd)-1):
    var_bylat[i] = np.nanmean(data['var'].sel(lat = slice(lat_bnd[i] - eps, lat_bnd[i+1] - eps)))
data.close()

fig, ax = plt.subplots(figsize = (4, 4))
ax.plot(0.5 * (lat_bnd[:-1] + lat_bnd[1:]), var_bylat, '-', color = 'k')
ax.set_xlabel('Lat')
ax.set_ylabel('Var [Unit]')
fig.savefig('myfig.png', dpi = 600., bbox_inches = 'tight')
plt.close(fig)

###############################################################################
# Calculate and plot mean time series of individual regions defined by levels 
# in a discrete mask.
###############################################################################


