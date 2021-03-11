import numpy as np

####
def _relative_entropy(vector, r_max):
    r"""
    Calculates the relative entropy of the climatology of precipitation.
    
    Parameters
    ----------
    vector: 1-d array
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
    The long-term seasonality index defined in
    
    Feng, X., A. Porporato, and I. Rodriguez-Iturbe, 2013: Changes in rainfall seasonality
    in the tropics. Nature Clim Change, 3, 811–815, https://doi.org/10.1038/nclimate1907.
    
    Meausures the evenness of the distribution of precipitation over months.
    Uses relative entropy and ranges between 0 and log_2(12) = 3.585, with larger
    values meaning that precipitation is more concentrated in the wet season.

    Parameters
    ----------
    vector: 1-d array
        12-element vector corresponding to a 12-month climatology.
    r_max: float
        whether to normalize the annual total by the global maximum in order to
        distinguish between different locations across the world
        (norm_frac = 7932 mm yr^{-1} at Tabubil, Papua New Guinea in the original paper).
    """
    D = _relative_entropy(vector, r_max)
    S = D * np.sum(vector) / r_max
    return S


def centroid(vector):
    r"""
    Measures the timing of rainfall using the first moment.

    Feng, X., A. Porporato, and I. Rodriguez-Iturbe, 2013: Changes in rainfall seasonality
    in the tropics. Nature Clim Change, 3, 811–815, https://doi.org/10.1038/nclimate1907.

    Parameters
    ----------
    vector: 1-d array
        12-element vector corresponding to a 12-month climatology.
    """"    
    C = np.sum( np.arange(1, 13) * vector ) / np.sum(vector)
    return C

def spread(vector, C = None):
    r"""
    Measures the duration of the wet season using the spread around the centroid.

    Feng, X., A. Porporato, and I. Rodriguez-Iturbe, 2013: Changes in rainfall seasonality
    in the tropics. Nature Clim Change, 3, 811–815, https://doi.org/10.1038/nclimate1907.

    Parameters
    ----------
    vector: 1-d array
        12-element vector corresponding to a 12-month climatology.
    """
    if C is None:
        C = centroid(vector)
    Z = np.sqrt( np.sum( np.power( np.arange(1, 13) - C, 2 ) * vector ) / np.sum(vector) )
    return Z
