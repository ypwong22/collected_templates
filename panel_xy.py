import numpy as np

####
from scipy.stats import gaussian_kde

def ax_scatter_density(ax, x, y, cmap = 'jet'):
    """Plot 2D scatter plot, colored by local density."""
    loc = np.vstack([x.ravel(), y.ravel()])
    f = gaussian_kde(loc)
    density = f(loc)
    h = ax.scatter(x, y, c = density, cmap = cmap,
                   edgecolor = '')
    return h
  
####
from scipy.stats import gaussian_kde

def ax_colored_density(ax, x, y, cmap = 'jet'):
    """Plot 2D Gaussian kde density plot."""
    loc = np.vstack([x.ravel(), y.ravel()])
    f = gaussian_kde(loc)

    x_range = np.linspace(np.min(x) - (np.max(x) - np.min(x))/10,
                          np.max(x) + (np.max(x) - np.min(x))/10, 100)
    y_range = np.linspace(np.min(y) - (np.max(y) - np.min(y))/10,
                          np.max(y) + (np.max(y) - np.min(y))/10, 100)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    z_mesh = f(np.vstack([x_mesh.ravel(), y_mesh.ravel()]))

    cf = ax.contourf(x_range, y_range, 
                     z_mesh.T.reshape(x_mesh.shape), cmap = cmap)
    return cf
