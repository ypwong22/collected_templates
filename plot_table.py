import matplotlib.pyplot as plt


def panel_table(ax, df):
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
