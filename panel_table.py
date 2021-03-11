import matplotlib.pyplot as plt
import numpy as np


def ax_table(ax, df):
    """
    Colored table according to values.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    norm = plt.Normalize(3, 6)
    colours = plt.cm.hot(norm(df.values))
    the_table = ax.table(cellText=df.values.astype(int),
                         rowLabels=df.index, colLabels=df.columns,
                         colWidths = [0.1]*vals.shape[1],
                         loc='center', fontsize = 16,
                         cellColours=colours)

    return the_table


def ax_stacked_bar(ax, x, matrix, color, edgecolor = None, **kwargs):
    """Stacked barplot with separate positive & negative values."""
    mat_pos = np.clip(matrix, 0., None)
    mat_pos_cum = np.cumsum(mat_pos, axis = 0)
    mat_neg = np.clip(matrix, None, 0.)
    mat_neg_cum = np.cumsum(mat_neg, axis = 0)

    if edgecolor == None:
        edgecolor = color

    h = [None] * len(color)
    for i in range(matrix.shape[0]):
        temp = mat_pos[i,:] # only plot positive values
        filt = np.abs(temp) > 1e-6
        if np.sum(filt) > 0:
            if i == 0:
                h[i] = ax.bar(x[filt], temp[filt], color = color[i],
                              edgecolor = edgecolor[i], **kwargs)
            else:
                h[i] = ax.bar(x[filt], temp[filt],
                              bottom = mat_pos_cum[i-1,:][filt],
                              color = color[i], edgecolor = edgecolor[i],
                              **kwargs)
        temp = mat_neg[i,:]
        filt = np.abs(temp) > 1e-6
        if np.sum(filt) > 0:
            if i == 0:
                ax.bar(x[filt], temp[filt], color = color[i],
                       edgecolor = edgecolor[i], **kwargs)
            else:
                ax.bar(x[filt], temp[filt],
                       bottom = mat_neg_cum[i-1,:][filt],
                       color = color[i], edgecolor = edgecolor[i], **kwargs)
    return h
