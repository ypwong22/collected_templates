###############################################################################
# Subset a global region using a boolean mask.
###############################################################################



###############################################################################
# Calculate and plot latitudinal mean time series.
###############################################################################
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

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


