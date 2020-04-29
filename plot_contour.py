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
mask = hr['biomes'].values[::-1, :].copy() # reverse S->N
keys = dict([(int(x.split('-')[0]), x.split('-')[1]) for x in hr['biomes'].attrs['biomes'].split('; ')])
keys_keys = list(keys.keys())
hr.close()

pct_mask = np.full([len(keys_keys), len(target_lat), len(target_lon)], np.nan)
for ind, k in enumerate(keys_keys):
    mask_0 = xr.DataArray(np.abs(mask - k) < 1e-8, 
                          coords = {'lat': source_lat, 'lon': source_lon},
                          dims = ['lat', 'lon'])
    pct_mask[ind, :, :] = mask_0.interp(lat = target_lat, lon = target_lon).values
# ---- fill by the maximum percentage area in each cell 
new_mask = np.argmax(pct_mask, axis = 0)
new_mask2 = np.full_like(new_mask, np.nan)
for ind, k in enumerate(keys_keys):
    new_mask2[np.abs(new_mask - ind) < 1e-8] = k

xr.DataArray(new_mask, coords = {'lat': target_lat, 'lon': target_lon},
             dims = ['lat', 'lon'], 
             attrs = '; '.join([str(i)+'-'+j for i,j in keys.items()]) \
            ).to_dataset(name = 'biomes')

fig, ax = plt.subplots(subplot_kw = {'projection': ccrs.Miller()})
cf = ax.contourf(target_lon, target_lat, new_mask, cmap = 'Spectral')
plt.colorbar(cf, ax = ax, orientation = 'horizontal')
fig.savefig('Biomes.png')
plt.close(fig)
