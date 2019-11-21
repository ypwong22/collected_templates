###############################################################################
# Calculate and plot the global mean time series.
###############################################################################
import xarray as xr
import matplotlib.pyplot as plt

fname = 'myfile.nc'
data = xr.open_dataset(fname, decode_times = True)
time = data['time'].indexes['time']
var_ts = data['var'].mean(dims = ['lat', 'lon'])
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


# Calculate the climatology.
fname = 'myfile.nc'
data = xr.open_dataset(fname, decode_times = True)
var = data['var'].mean(dims = ['time'])
data.close()


# Create the figure, get the panel (ax)
...

# Options - Change here
cmap = 'Spectral'
levels = np.linspace(-1., 1., 10)
map_extent = [-180, 180, -60, 70]
grid_on = True # True, False


# Generic module: var - some xr.DataArray
ax.coastlines()
ax.set_extent(map_extent)
h = ax.contourf(var.lat, var.lon, var, cmap = cmap, levels = levels)
plt.colorbar(h, ax = ax, boundaries = levels)
if grid_on:
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 20.))
    gl.ylocator = mticker.FixedLocator(np.arange(-90., 90., 10.))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'black', 'weight': 'bold', 'size': 10}
    gl.ylabel_style = {'color': 'black', 'weight': 'bold', 'size': 10}
