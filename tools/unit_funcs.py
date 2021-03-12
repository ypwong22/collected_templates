import numpy as np

####
def _relative_entropy(vector, r_max):
    r"""
    Calculates the relative entropy of the climatology of precipitation.
    
    Parameters
    ----------
    vector: 1-d array
        12-element vector corresponding to a 12-month climatology of the hydrological year.
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
        12-element vector corresponding to a 12-month climatology of the hydrological year.
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
        12-element vector corresponding to a 12-month climatology of the hydrological year.
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
        12-element vector corresponding to a 12-month climatology of the hydrological year.
    """
    if C is None:
        C = centroid(vector)
    Z = np.sqrt( np.sum( np.power( np.arange(1, 13) - C, 2 ) * vector ) / np.sum(vector) )
    return Z


def entropic_spread(vector):
    r"""
    An information theory-based measure for the support of the monthly rainfall distribution
    in each year. Therefore, it is a measure of the duration of the rainy season and is
    defined based on the information entropy of each year.
    
    Analogous to spread.
    
    Feng, X., A. Porporato, and I. Rodriguez-Iturbe, 2013: Changes in rainfall seasonality
    in the tropics. Nature Clim Change, 3, 811–815, https://doi.org/10.1038/nclimate1907.

    Parameters
    ----------
    vector: 1-d array
        12-element vector corresponding to a 12-month climatology of the hydrological year.
    """
    frac_month = vector / np.sum(vector)
    H = - np.sum( frac_month * np.log2(frac_month) ) # information entropy
    E = np.sqrt( (np.power(2, 2 * H) - 1) / 12 )
    return E


def demodulated_amplitude_n_phase(time_series):
    r"""
    The demodulated amplitude is akin to mean, and the demodulated phase is
    akin to centroid. But they are derived based on localized harmonic analysis. 

    The time series was multiplied by two sinusoidal functions, a low pass filter 
    applied to isolate the low frequencies from the high frequencies, and 
    the amplitude and phase of the low frequency component identified. 
    
    Feng, X., A. Porporato, and I. Rodriguez-Iturbe, 2013: Changes in rainfall seasonality
    in the tropics. Nature Clim Change, 3, 811–815, https://doi.org/10.1038/nclimate1907.

    Parameters
    ----------
    time_series: 1-d array
        Time series of, e.g., monthly precipitation. Must be a multiple of 12.
    """
    T = 12

    # Transform
    y = time_series * np.exp(-1j * 2 * np.pi / T * np.arange(1, len(time_series) + 1))
    
    # Hanning window
    weight = np.append(np.insert(np.ones(11), 0, 0.5), 0.5)
    F = np.append(np.insert( 1/T * np.convolve(y, weight, mode = 'valid'), 0, 
                             np.full(T//2, np.nan) ), np.full(T//2, np.nan))
    A = 2 * np.sqrt( np.power(F.real, 2) + np.power(F.imag, 2) )
    theta = np.arctan( F.imag / F.real )
    
    # Resample to annual
    A = A[::12]
    theta = theta[::12] / np.pi / 2 * T # convert from angular frequency to period
    
    return A, theta
