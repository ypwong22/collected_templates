# Various matplotlib, cartopy, colorbar tricks that I learned today

1. **Add subplot to matplotlib figure**

```python
fig.add_subplot(abc, position = [left, bottom, width, height])
```

The three numbers refer to a – the number of rows, b – the number of columns, c – the position of the subplot in the grid. The ``position =`` argument controls the anchor of this subplot. 

Another way of having control over the subplots is to pass in a **GridSpec** object:

The three numbers refer to *a* – the number of rows, *b* – the number of columns, *c* – the position of the subplot in the grid. The ``position =`` argument controls the anchor of this subplot. 

Another way of having control over the subplots is to pass in a **GridSpec** object:

```python
fig = plt.figure()
gs = fig.add_gridspec(5, 5)
ax1 = fig.add_subplot(gs[:2, :])
ax2 = fig.add_subplot(gs[2:4, :])
# last subplot is smaller
ax3 = fig.add_subplot(gs[4, 2:3])
```



2. **Using cartopy to plot global maps at different projections, one first specifies the desired projection of the map by passing the ``projection =`` argument to subplot, e.g.**

```python
import cartopy.crs as ccrs
plt.subplots(nrows=2, ncols=1, figsize=(12,12), subplot_kw={'projection': ccrs.EqualEarth()})
```

or

```python
fig.add_subplot(gs[:2,:], projection={'projection': ccrs.EqualEarth()})
```

But, when using **contour()/contourf()/pcolormesh()**, the ```transform=``` argument is NOT the desired projection of the map. Instead, it is the actual transformation of the X, Y, Z data. In the case of geographical coordinates, one should use the **PlateCarree()** or **RotatedPole()** projection, i.e. 

```python
ax.contourf(data.lon, data.lat, data.var, transform=ccrs.PlateCarree(), cmap='RdYlBu_r')
```

 

3. **More about using cartopy - if the global data is not cyclic, using the contourf() function as in 2. will result in a white strip at lon=0. Therefore, one needs to add the cyclic point before plotting, i.e.**

```python
from cartopy.util import add_cyclic_point
var_cyc, lon_cyc = add_cyclic_point(data.var, coord=data.lon)
ax.contourf(lon_cyc, data.lat, var_cyc, transform=ccrs.PlateCarree(), cmap='RdYlBu_r')
```



4. **More about using cartopy**

To display a global map instead of having the map clipped to only where there is data, use

``ax.set_global()``

Also, one can add coastlines by using

``ax.coastlines()``

 

5. **When using contourf(), if one want to have a nice-looking colorbar that is not clipped to the actual range of the data, it is necessary to manually specify the contour levels using the ``levels=`` argument, e.g.** 

```python
ax.contourf(lon_cyc, data.lat, var_cyc, levels = np.linspace(-35, 35, 10), transform=ccrs.PlateCarree(), cmap='RdYlBu_r')
```

 

6. **More about using contourf() (jointly with colorbar()). If one wants to manually specify the colorbar levels, one need to pass things into the “levels=” and “norm=” arguments of the contourf() function, and into the “boundaries =” and “norm=” arguments of the colorbar() function. For example:** 

```python
from matplotlib import colors
bounds = np.array([1,3,5,6,8,10,355])
norm = colors.BoundaryNorm(bounds, 256) # No need to think why but just use 256 in the second argument.
cf = ax.contourf(lon_cyc, data.lat, var_cyc, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', levels = bounds, norm = norm)
cbar = fig.colorbar(mappable = cf, ax= ax, orientation='horizontal', norm=norm, boundaries = bounds)
```

7. **To control the ticks of the colorbar, one can manually specify ticks using the ``tick =`` argument to the colorbar() function. To control the ticklabels, one need to pass a tickformatter function into the cbar.ax.xaxis.set_major_formatter() function of the axis of the colorbar. It is convenient to use the matplotlib.ticker.FixedFormatter() function to simply specify the labels manually.** 

```python
from matplotlib.ticker import FixedFormatter
cbar = fig.colorbar(mappable = cf, ax= ax, orientation='horizontal', norm=norm, boundaries = bounds, ticks = bounds)
cbar.ax.xaxis.set_major_formatter(FixedFormatter(['%g %' x for x in bounds]) # the %g formatter removes trailing zeros in decimal numbers
```

8. **Other tricks to control colorbar() behavior:** 

Control the distance between the colorbar and the subplot: ``fig.colorbar(…, pad=0.05, …)``

Reduce the size of the colorbar relative to the figure and change the width:height ratio: ``fig.colorbar(…, shrink=0.5, aspect=20 …)``

 

 