def fit_piecewise_linear(x, y, nbreaks):
    """
    Fit n-piecewise linear regression equations between x and y, return parameters. 

    Modified from https://datascience.stackexchange.com/questions/8457/python-library-for-segmented-regression-a-k-a-piecewise-regression 

    Test cases

    # 2 breakpoints case
    x = np.arange(-100, 100.)
    np.random.randn(44)
    y = np.piecewise(x, [x < 10, (x >= 10) & (x < 50), x >= 50], 
                    [lambda x: -10 - 5*x, 
                    lambda x: -10 - 5*10 + 2.5*(x-10), 
                    lambda x: -10 - 5*10 + 2.5*(50-10) + 5*(x-50)]) + np.random.randn(len(x)) * 3
    y0, ey0, x_breaks, ex_breaks, k_breaks, pk_breaks, rmse, func = fit_piecewise_linear(x, y, 2)

    fig, ax = plt.subplots()
    ax.plot(x, y, 'ok', markersize = 1)
    ax.plot(x, func(x, y0, *x_breaks, *k_breaks), '-b')
    fig.savefig(os.path.join(path_out, 'test.png'))
    print(rmse)

    # 3 breakpoints case
    x = np.arange(-40, 100.)
    np.random.randn(44)
    y = np.piecewise(x, [x < 10, (x >= 10) & (x < 30), (x >= 30) & (x < 60), x >= 60], 
                    [lambda x: -10 - 5*x, 
                    lambda x: -10 - 5*10 + 2.5*( x-10), 
                    lambda x: -10 - 5*10 + 2.5*(30-10) + 0*( x-30),
                    lambda x: -10 - 5*10 + 2.5*(30-10) + 0*(60-30) + 5*(x-60)]) + \
        np.random.randn(len(x)) * 3
    y0, ey0, x_breaks, ex_breaks, k_breaks, pk_breaks, rmse, func = fit_piecewise_linear(x, y, 3)

    fig, ax = plt.subplots()
    ax.plot(x, y, 'ok', markersize = 1)
    ax.plot(x, func(x, y0, *x_breaks, *k_breaks), '-b')
    fig.savefig(os.path.join(path_out, 'test1.png'))
    print(rmse)
    """

    if nbreaks < 1:
        raise "Cannot handle zero break situations"

    def make_func(nbreaks):
        """ Test case: 

        func = make_func(1)
        x = np.arange(-10, 10).astype(float)
        y = func(x, 1, 5, -1, 3)

        func2 = make_func(2)
        y2 = func2(x, 2, 2, 6, 0, 2, 9)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.plot(x, y2)
        fig.savefig(os.path.join(path_out, 'test.png'))
        """

        def piecewise_linear(x, *args):
            """ y0      : the intercept at x = 0
                x_breaks: the inflection points
                k_breaks: the slopes before and after the inflection points """
            y0       = args[0]
            x_breaks = np.array(args[1:(nbreaks+1)])
            k_breaks = np.array(args[(nbreaks+1):])
            if len(x_breaks) != (len(k_breaks)-1):
                raise "The number breakpoints must be equal to the number of breakpoints -1."
            # ensure the x_breaks are from smallest to largest
            order    = np.argsort(x_breaks)
            x_breaks = x_breaks[order]
            k_breaks = k_breaks[np.insert(order + 1, 0, 0)]

            x_breaks = np.insert(x_breaks, 0, 0)
            segments = []
            funcs    = []
            for i in range(len(x_breaks)):
                if i == 0:
                    seg = x < x_breaks[1]
                elif i == (len(x_breaks)-1):
                    seg = x >= x_breaks[i]
                else:
                    seg = (x >= x_breaks[i]) & (x < x_breaks[i+1])

                # use scope to prevent intercept & slope being changed before lambda is called
                intercept = y0 + np.sum(k_breaks[:i] * (x_breaks[1:(i+1)] - x_breaks[:i])) - \
                            k_breaks[i] * x_breaks[i]
                slope     = k_breaks[i]
                def newfunc(x, intercept, slope):
                    return lambda x: intercept + slope * x

                segments.append(seg)
                funcs   .append(newfunc(x, intercept, slope))
            return np.piecewise(x, segments, funcs)

        return piecewise_linear

    func      = make_func         (nbreaks)
    p , e     = optimize.curve_fit(func, x, y, np.zeros(2*nbreaks+2))
    y_        = func              (x.astype(float) , *p)

    y0        = p[0]
    x_breaks  = p[1:(nbreaks+1)]
    k_breaks  = p[(nbreaks+1):]

    ey0       = np.sqrt(np.diag(e)[0])
    ex_breaks = np.sqrt(np.diag(e)[1:(nbreaks+1)])
    ek_breaks = np.sqrt(np.diag(e)[(nbreaks+1):])

    # re-order x_breaks from smallest to largest
    order    = np.argsort(x_breaks)
    x_breaks = x_breaks[order]
    ex_breaks = ex_breaks[order]
    k_breaks = k_breaks[np.insert(order + 1, 0, 0)]
    ek_breaks = ek_breaks[np.insert(order + 1, 0, 0)]

    pk_breaks = 2 * t.sf(np.abs(k_breaks/ek_breaks), len(x)-2*nbreaks)
    rmse     = np.sqrt(np.mean(np.power(y - y_, 2)))

    return y0, ey0, x_breaks, ex_breaks, k_breaks, pk_breaks, rmse, func
