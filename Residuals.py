from lmfit.lineshapes import lorentzian, gaussian, split_lorentzian


def residuals_lorentzian(parameters, x, y):
    """
    A function which calculates the residuals between a set of fitted data points and the original data points.
    This function will be called using the lmfit Minimizer object as an objective function to be minimized.

    This residual function calculates the residuals based on fitting a Lorentizan lineshape to the single peak.

    Parameter object must first be initialized and the parameters within the object must be pre-defined using
    a best guess.

    :param parameters: An lmfit Parameters object defined by the user.
    :param x: Numpy array containing the x-values.
    :param y: Numpy array containing the y-values.

    :return: residuals - Numpy array containing the residuals between the fitted model’s
                         y-values and the actual y-values.
    """
    model = lorentzian(x, parameters['p1_amplitude'], parameters['p1_center'], parameters['p1_half_width'])
    residuals = model - y
    return residuals


def residuals_gaussian(parameters, x, y):
    """
    A function which calculates the residuals between a set of fitted data points and the original data points.
    This function will be called using the lmfit Minimizer object as an objective function to be minimized.

    This residual function calculates the residuals based on fitting a Gaussian lineshape to the single peak.

    Parameter object must first be initialized and the parameters within the object must be pre-defined using
    a best guess.

    :param parameters: An lmfit Parameters object defined by the user.
    :param x: Numpy array containing the x-values.
    :param y: Numpy array containing the y-values.

    :return: residuals - Numpy array containing the residuals between the fitted model’s
                         y-values and the actual y-values.
    """
    model = gaussian(x, parameters['p1_amplitude'], parameters['p1_center'], parameters['p1_half_width'])
    residuals = model - y
    return residuals


def residuals_vinyl(parameters, x, y):
    """
    A function which produces the residuals between a set of fitted data points and the original data points.
    This function will be called using the lmfit Minimizer object as an objective function to be minimized.

    This objective function is appropriate when there are two convoluted peaks which can be fitted to a linear
    combination of 2 Lorentizan lineshapes. This objective function was specifically used to fit the vinyl region.

    Parameter object must first be initialized and the parameters within the object must be pre-defined using
    a best guess.

    :param parameters: An lmfit Parameters object defined by the user.
    :param x: Numpy array containing the x-values.
    :param y: Numpy array containing the y-values.

    :return: residuals - Numpy array containing the residuals between the fitted model’s
                         y-values and the actual y-values.
    """
    model = (lorentzian(x, parameters['p1amp'], parameters['p1center'], parameters['p1width']) +
             lorentzian(x, parameters['p2amp'], parameters['p2center'], parameters['p2width']))
    residuals = model - y
    return residuals


def residuals_pxylene(parameters, x, y):
    """
    A function which produces the residuals between a set of fitted data points and the original data points.
    This function will be called using the lmfit Minimizer object as an objective function to be minimized.

    This objective function is appropriate when there is a peak to the left and two convoluted peaks to the right,
    which can be fitted to a linear combination of a Lorentizan, a second Lorentzian and a Split-Lorentzian lineshape.
    This objective function was specifically used to fit the p-xylene region.

    Parameter object must first be initialized and the parameters within the object must be pre-defined using
    a best guess.

    :param parameters: An lmfit Parameters object defined by the user.
    :param x: Numpy array containing the x-values.
    :param y: Numpy array containing the y-values.

    :return: residuals - Numpy array containing the residuals between the fitted model’s
                         y-values and the actual y-values.
    """
    model = (lorentzian(x, parameters['p1amp'], parameters['p1center'], parameters['p1width']) +
             lorentzian(x, parameters['p2amp'], parameters['p2center'], parameters['p2width']) +
             split_lorentzian(x, parameters['p3amp'], parameters['p3center'], parameters['p3width_left'],
                              parameters['p3width_right']))
    residuals = model - y
    return residuals
