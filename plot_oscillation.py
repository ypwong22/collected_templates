# A script to generate the common blue-red style ocean indices plot
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl


inx = pd.read_csv('inx.csv', index_col = 0)


mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.titlesize'] = 6
mpl.rcParams['axes.titleweight'] = 'bold'


fig, axes = plt.subplots(5, 1, figsize = (5, 5), sharex = True,
                         sharey = False)
for ind, name in enumerate(['AMO', 'IDMI', 'NAO', 'PDO', 'SOI']):
    ax = axes[ind]
    ax.fill_between(inx.index.values, inx[name].values, 0,
                    where = inx[name].values >= 0, color = 'r',
                    interpolate = True)
    ax.fill_between(inx.index.values, inx[name].values, 0,
                    where = inx[name].values <= 0, color = 'b',
                    interpolate = True)
    ax.plot(inx.index.values, inx[name].values, '-k', lw = 0.5)
    ax.set_title(name, pad = 0.5)
    ax.axvline(2012, ls = '-', color = 'k')
    ax.set_xlim([1900, 2015])
    ax.set_xticks(np.arange(1912, 2013, 20))

fig.savefig('inx.png', dpi = 600., bbox_inches = 'tight')
plt.close(fig)
