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
eps = 0.01 # floating point tolerance

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
lat_bnd_median = 0.5 * (lat_bnd[:-1] + lat_bnd[1:])
eps = 0.01 # Tolerance for floating point latitude comparison.

data = xr.open_dataset('myfile.nc')

var_bylat = np.full([len(data['time']), len(lat_bnd_median)], np.nan)

# use cosine of latitude as surrogate for grid cell size, or read from a file.
lat_bnd_wgts = np.cos(np.deg2rad(data.lat.values))

for i in range(len(lat_bnd_median)):
    # like pandas, label based indexing in xarray is inclusive of both the start and the stop bounds    
    var_temp = data['var'].sel(lat = slice(lat_bnd[i] - eps, 
                                           lat_bnd[i+1] - eps)).mean(dim = 'lon').values.reshape(-1)

    # if multiple latitudes exist, use the appropriate weight to average over:
    # (assume time is the first dimension, latitude is the second dimension)
    if (len(var_temp.shape) > 1) and (var_temp.shape[1] > 1):
        wgts_temp = lat_bnd_wgts[(data.lat.values >= lat_bnd[i]) & \
                                 (data.lat.values < lat_bnd[i+1])].copy()
        wgts_temp[np.isnan(var_temp[0, :])] = np.nan
        wgts_temp = wgts_temp / np.nanmean(wgts_temp)

        var_bylat[:, i] = np.nanmean(var_temp * wgts_temp, axis = 1)
    else:
       var_bylat[:, i] = varp_temp.reshape(-1)

data.close()

fig, ax = plt.subplots(figsize = (4, 4))
ax.plot(lat_bnd_median, var_bylat.mean(axis = 0), '-', color = 'k')
ax.set_xlabel('Lat')
ax.set_ylabel('Var [Unit]')
fig.savefig('myfig.png', dpi = 600., bbox_inches = 'tight')
plt.close(fig)


###############################################################################
# Calculate and plot longitudinal mean time series. Should provide the weight
# from grid sizes.
###############################################################################
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

lon_bnd = np.arange(-180., 180., 10.)
lon_bnd_median = 0.5 * (lon_bnd[:-1] + lon_bnd[1:])
eps = 0.01 # Tolerance for floating point latitude comparison.

data = xr.open_dataset('myfile.nc')

var_bylon = np.full([len(data['time']), len(lon_bnd_median)], np.nan)

# use cosine of latitude as surrogate for grid cell size, or read from a file.
lat_bnd_wgts = np.cos(np.deg2rad(data.lat.values))

for i in range(len(lon_bnd)-1):
    # like pandas, label based indexing in xarray is inclusive of both the start and the stop bounds
    var_temp = data['var'].sel(lon = slice(lon_bnd[i] - eps, 
                                           lon_bnd[i+1] - eps)).mean(dim = 'lon').values

    # normalize the weights according to the number of grid cells that have valid values.
    wgts_temp = lat_bnd_wgts.copy()
    wgts_temp[np.isnan(var_temp[0, :])] = np.nan # assume time is the first dimension
    wgts_temp = wgts_temp / np.nanmean(wgts_temp)

    var_bylon[:, i] = np.nanmean(var_temp * wgts_temp, axis = 1) # lat is the second dimension

data.close()

fig, ax = plt.subplots(figsize = (4, 4))
ax.plot(lon_bnd_median, var_bylon.mean(axis = 0), '-', color = 'k')
ax.set_xlabel('Lon')
ax.set_ylabel('Var [Unit]')
fig.savefig('myfig.png', dpi = 600., bbox_inches = 'tight')
plt.close(fig)


###############################################################################
# Calculate and plot mean time series of individual regions defined by levels 
# in a discrete mask.
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

# Get land area for weighting purpose.
data1 = xr.open_dataset('land_area.nc')
area = data1['area'].values.copy()
data1.close()

# Get the discrete values of the mask.
mask_levels = np.unique(mask[~np.isnan(mask)])

# Get data and apply mask.
data = xr.open_dataset('mydata.nc')
eps = 1e-6 # floating point number tolerance
var = data['var'].values.copy()

var_bymask = np.full([len(data['time']), len(mask_levels)], np.nan)

for mk_ind, mk in enumerate(mask_levels):
    var_temp = np.where(np.abs(mask - mk) < eps, var, np.nan)
    area_temp = np.where(np.abs(mask - mk) < eps, area, np.nan)

    # Keep the time dimension
    var_bymask[:, mk_ind] = np.nanmean(np.nanmean(var_temp * area_temp, 
                                                  axis = 2), axis = 1) / np.nanmean(area_temp)

data.close()

# Accompanying plot.
fig, ax = plt.subplots()
ax.plot(data['time'].values, var_bymask)
fig.savefig('myfig.png')
plt.close(fig)


###############################################################################
# Plot regional masks by different colors.
###############################################################################
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import cm
import numpy as np

cmap = cm.get_cmap('Spectral')
map_extent = [-180, 180, -60, 90]
grid_on = True # True, False
colorbar = True # True, False
cm_label = True # True, False; label the level at center of mass
eps = 1e-6

# Get mask: True - retain values, False - discard values.
data0 = xr.open_dataset('mask.nc')
mask = data0['mask'].values.copy()
data0.close()

mask_levels = np.sort(np.unique(mask[~np.isnan(mask)]))
mask_levels_bounds = np.append(mask_levels - (mask_levels[1]-mask_levels[0])/2,
                               [mask_levels[-1] + (mask_levels[-1]-mask_levels[-2])/2])

... # create figure and ax

# Plot.
ax.coastlines()
ax.set_extent(map_extent)
mask_cyc, lon_cyc = add_cyclic_point(mask, coord=data0.lon)
cf = ax.contourf(lon_cyc, data0.lat, mask_cyc, cmap = cmap, levels = mask_levels_bounds)

if colorbar:
    cb = plt.colorbar(cf, ax = ax, boundaries = mask_levels_bounds, shrink = 0.7)
    cb.ax.set_yticks(mask_levels)

if cm_label:
    lon2d, lat2d = np.meshgrid(data0.lon.values, data0.lat.values)
    for mk_ind, mk in enumerate(mask_levels):
        mask_subset = (np.abs(mask - mk) < eps).astype(int)
        x = np.average(lon2d, weights = mask_subset)
        y = np.average(lat2d, weights = mask_subset)
        ax.text(x, y, '%d' % mk, fontdict = {'fontsize': 12})

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
ax.text(-0.07, 0.55, 'latitude', va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',
        transform=ax.transAxes)
ax.text(0.5, -0.2, 'longitude', va='bottom', ha='center',
        rotation='horizontal', rotation_mode='anchor',
        transform=ax.transAxes)
