def circular_linear(y, x, T, zero):
    """
    Copied from the "circular" R package Version 0.4-7 2013/11/06
    
    Ulric Lund <ulund@calpoly.edu> from "Topics in circular Statistics" (2001) S. Rao Jammalamadaka and A. SenGupta, World Scientific
    Claudio Agostinelli <claudio@unive.it>
    Ulric Lund <ulund@calpoly.edu>
    https://github.com/cran/circular

    Parameters
    ----------
    y: 1-d numpy array
      A circular variable with period given by T, and zero-value given by zero.
    x: 1-d numpy array
      A non-circular variable.
    """
