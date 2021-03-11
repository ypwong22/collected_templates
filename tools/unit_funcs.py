import numpy as np

####
def relative_entropy(vector, norm_frac = None):
    r"""
    Meausures the evenness of the distribution of precipitation over months.
    Larger values means precipitation is more concentrated in the wet season.
    
    Parameters
    ----------
    vector: ndarray
        12-element vector corresponding to a 12-month climatology.
    norm_frac: float
        whether to normalize the annual total by the global maximum in order to
        distinguish between different locations across the world
        (norm_frac = 7932 mm yr^{-1} at Tabubil, Papua New Guinea in the original paper).

    Feng, X., A. Porporato, and I. Rodriguez-Iturbe, 2013: Changes in rainfall seasonality
    in the tropics. Nature Clim Change, 3, 811–815, https://doi.org/10.1038/nclimate1907.
    """
    frac_month = vector / np.sum(vector)
    if not (norm_frac is None):
        frac_month *= norm_frac
    d = np.sum(frac_month * np.log2(frac_month * 12))
    return d
