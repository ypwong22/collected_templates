"""
A visualization of the US population trend by state.

Source of population data: US Census Bureau.
"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib.cm import get_cmap, ScalarMappable
import shapefile as shp
import numpy as np


def read_file():
    pt = pd.read_excel("nst-est2019-01.xlsx", sheet_name = "NST01",
                       skiprows = 9, header = None,
                       index_col = 0, usecols = 'A,N')
    pt = pt.iloc[:51, :]
    pt.index = [x.replace('.', '') for x in pt.index]
    return pt


def assign_colors(dfs):
    vector = dfs.values.reshape(-1)
    names = dfs.index
    cmap = get_cmap('Spectral')
    norm = Normalize(np.min(vector), np.max(vector))
    clist = dict([(names[i], cmap(norm(vector[i]))) \
                  for i in range(len(vector))])
    return clist, cmap, norm


def plot_geom(ax, geoms, clist):
    count = 0
    for gm in geoms:
        if gm.record.NAME in clist:
            ax.add_patch(Polygon(gm.shape.points, fc = clist[gm.record.NAME], 
                                 lw = 0.5, ec = 'k'))
        count += 1
    return ax

def add_colorbar(ax, cmap, norm):
    sm = ScalarMappable(cmap = cmap, norm = norm)
    sm.set_clim(norm.vmin, norm.vmax)
    plt.colorbar(sm, ax = ax)

if __name__ == '__main__':
    pt = read_file()
    clist, cmap, norm = assign_colors(pt)

    fig, ax = plt.subplots(figsize = (8, 6),
                           subplot_kw = {'projection': ccrs.PlateCarree()})
    ax.set_extent([-126.5, -68.5, 25.5, 50.5])    
    sf = shp.Reader('cb_2018_us_state_5m/cb_2018_us_state_5m.shp')
    plot_geom(ax, sf.shapeRecords(), clist)
    add_colorbar(ax, cmap, norm)
    sf.close()

    fig.savefig('nst-est2019-01.png', dpi = 600, bbox_inches = 'tight')
    plt.close(fig)