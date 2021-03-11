import numpy as np

####
def _relative_entropy(vector, r_max):
    r"""
    Calculates the relative entropy of the climatology of precipitation.
    
    Parameters
    ----------
    vector: ndarray
        12-element vector corresponding to a 12-month climatology.
    r_max: float
        whether to normalize the annual total by the global maximum in order to
        distinguish between different locations across the world
        (norm_frac = 7932 mm yr^{-1} at Tabubil, Papua New Guinea in the original paper).

    Feng, X., A. Porporato, and I. Rodriguez-Iturbe, 2013: Changes in rainfall seasonality
    in the tropics. Nature Clim Change, 3, 811–815, https://doi.org/10.1038/nclimate1907.
    """
    frac_month = vector / np.sum(vector)
    if not (r_max is None):
        frac_month *= r_max
    D = np.sum(frac_month * np.log2(frac_month * 12))
    return D

def longterm_seasonality(vector, r_max):
    r"""
    Uses relative entropy. Ranges between (0, log_2(12) = 3.585)
    Meausures the evenness of the distribution of precipitation over months.
    Larger values means precipitation is more concentrated in the wet season.

    Parameters
    ----------
    vector: ndarray
        12-element vector corresponding to a 12-month climatology.
    r_max: float
        whether to normalize the annual total by the global maximum in order to
        distinguish between different locations across the world
        (norm_frac = 7932 mm yr^{-1} at Tabubil, Papua New Guinea in the original paper).

    Feng, X., A. Porporato, and I. Rodriguez-Iturbe, 2013: Changes in rainfall seasonality
    in the tropics. Nature Clim Change, 3, 811–815, https://doi.org/10.1038/nclimate1907.
    """
    D = _relative_entropy(vector, r_max)
    S = D * np.sum(vector) / r_max
    return S
