############################################
# Calculate and plot the global mean
############################################
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
