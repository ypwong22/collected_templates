###############################################################################
# Interpolate and plot NetCDF file of multi-region masks
###############################################################################
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

hr = xr.open_dataset('Biomes.nc')
keys = dict([(int(x.split('-')[0]), x.split('-')[1]) for x in hr['biomes'].attrs['biomes'].split('; ')])

hr.close()
