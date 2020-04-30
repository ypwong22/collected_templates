###############################################################################
# Interpolate and plot NetCDF file of multi-region masks
###############################################################################
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

target_lat = ... # numpy array
target_lon = ... # numpy array

hr = xr.open_dataset('Biomes.nc')
source_lat = hr.lat.values[::-1] # reverse S->N
source_lon = hr.lon.values # already
# ---- do not mask invalid values. Otherwise the interpolated land mask
#      will be shrunk.
mask = hr['biomes'].values[::-1, :].copy() # reverse S->N
keys = dict([(int(x.split('-')[0]), x.split('-')[1]) \
             for x in hr['biomes'].attrs['long_name'].split('; ')[:-1]])
keys_keys = list(keys.keys())
hr.close()

pct_mask = np.full([len(keys_keys), len(target_lat), len(target_lon)], np.nan)
for ind, k in enumerate(keys_keys):
    mask_0 = xr.DataArray((np.abs(mask - k) < 1e-8).astype(float),
                          coords = {'lat': source_lat, 'lon': source_lon},
                          dims = ['lat', 'lon'])
    pct_mask[ind, :, :] = mask_0.interp(lat = target_lat,
                                        lon = target_lon).values
# ---- fill by the maximum percentage area in each cell
new_mask = np.argmax(pct_mask, axis = 0)
new_mask2 = np.full(new_mask.shape, np.nan)
for ind, k in enumerate(keys_keys):
    new_mask2[np.abs(new_mask - ind) < 1e-8] = float(k)
new_mask2[ np.isnan(np.max(pct_mask, axis = 0)) | \
           np.abs(np.max(pct_mask, axis = 0) < 1e-6) ] = np.nan

xr.DataArray(new_mask2, coords = {'lat': target_lat, 'lon': target_lon},
             dims = ['lat', 'lon'],
             attrs = {'keys': '; '.join([str(i)+'-'+j \
                                         for i,j in keys.items()])} \
).to_dataset(name = 'biomes').to_netcdf('Biomes_interp.nc')

fig, ax = plt.subplots(subplot_kw = {'projection': ccrs.Miller()})
ax.coastlines(lw = 0.1)
cf = ax.contourf(target_lon, target_lat, new_mask2, cmap = 'jet',
                 transform = ccrs.PlateCarree(),
                 levels = [0.5] + [0.5 + x for x in keys_keys])
hb = plt.colorbar(cf, ax = ax)
hb.set_ticks(keys_keys)
hb.set_ticklabels([keys[i] for i in keys_keys])
fig.savefig('Biomes_interp.png', dpi = 600., bbox_inches = 'tight')
plt.close(fig)
