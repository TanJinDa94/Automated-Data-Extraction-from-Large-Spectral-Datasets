from lmfit import Minimizer
from scipy import integrate
from sklearn.metrics import r2_score
import numpy as np
from lmfit.lineshapes import lorentzian, split_lorentzian


def lorentzian_curve_fit(residuals, parameters, x, y):
    """
    Fit the peak via given a specified objective function (the residual which takes into account the lineshape), a set
    of parameters that is required to fit the lineshape, and the peak containing region's x-values and y-values.

    :param residuals: Function imported from Residuals module which is the objective function to be minimized.
    :param parameters: Parameter Object which contains all the relevant parameters for curve fitting.
    :param x: Numpy array of x-values
    :param y: Numpy array of y-values

    :return: best_fit - Numpy array containing the y-values of the best fit lineshape for the peak
             fit_params - Ordered dictionary of best fit parameters that can best fit the data, including the following:
             1. Amplitude
             2. Center
             3. Half-Width at Half-Maximum
             4. R-squared
             5. FWHM
             6. Height
             7. AUC
    """
    mini = Minimizer(residuals, parameters, fcn_args=(x, y))  # Initialize Minimizer object
    out = mini.leastsq()  # Use Levenberg-Marquardt minimization to perform a fit.
    best_fit = y + out.residual
    # out.residual is a Numpy array of the minimized objective function when using the best-fit values
    # of the parameters. The best fit curve is therefore the y values plus the minimized residuals.
    # best_fit is also a Numpy array.

    fit_params = out.params.valuesdict()  # Returns an ordered dictionary of parameter values.

    fit_params['r2_score'] = r2_score(y, best_fit)
    fit_params['fwhm'] = 2 * fit_params['p1_half_width']
    fit_params['height'] = (1 / np.pi) * fit_params['p1_amplitude'] / max(np.finfo(float).eps,
                                                                          fit_params['p1_half_width'])
    fit_params['auc'] = integrate.simpson(best_fit, x)  # Integrate the area below best_fit to get the AUC.
    # Height and FWHM equations are written according to the functional form in the documentation.
    # np.finfo(float).eps is the non-zero value of machine limit for floating points. Non-zero value is used so that
    # we do not divide by zero. eps = 2**-52, approximately 2.22e-16.

    return best_fit, fit_params


def gaussian_curve_fit(residuals, parameters, x, y):
    """
    Fit the peak via given a specified objective function (the residual which takes into account the lineshape), a set
    of parameters that is required to fit the lineshape, and the peak containing region's x-values and y-values.

    :param residuals: Function imported from Residuals module which is the objective function to be minimized.
    :param parameters: Parameter Object which contains all the relevant parameters for curve fitting.
    :param x: Numpy array of x-values
    :param y: Numpy array of y-values

    :return: best_fit - Numpy array containing the y-values of the best fit lineshape for the peak
             fit_params - Ordered dictionary of best fit parameters that can best fit the data, including the following:
             1. Amplitude
             2. Center
             3. Half-Width at Half-Maximum
             4. R-squared
             5. FWHM
             6. Height
             7. AUC
    """
    mini = Minimizer(residuals, parameters, fcn_args=(x, y))  # Initialize Minimizer object
    out = mini.leastsq()  # Use Levenberg-Marquardt minimization to perform a fit.
    best_fit = y + out.residual
    # out.residual is a Numpy array of the minimized objective function when using the best-fit values
    # of the parameters. The best fit curve is therefore the y values plus the minimized residuals.
    # best_fit is also a Numpy array.

    fit_params = out.params.valuesdict()  # Returns an ordered dictionary of parameter values.

    fit_params['r2_score'] = r2_score(y, best_fit)  # Computes the r2 score between the best_fit and y.
    fit_params['fwhm'] = 2 * ((2 * np.log(2)) ** 0.5) * fit_params['p1_half_width']
    fit_params['height'] = (1 / (2 * np.pi) ** 0.5) * fit_params['p1_amplitude'] / max(np.finfo(float).eps,
                                                                                       fit_params['p1_half_width'])
    fit_params['auc'] = integrate.simpson(best_fit, x)  # Integrate the area below best_fit to get the AUC.

    # Height and FWHM equations are written according to the functional form in the documentation.
    # np.finfo(float).eps is the non-zero value of machine limit for floating points. Non-zero value is used so that
    # we do not divide by zero. eps = 2**-52, approximately 2.22e-16.

    return best_fit, fit_params


def curve_fit(residuals, parameters, x, y, region):
    """
    Fit a curve to the region of interest. This curve fitting function was specifically written for the vinyl
    and p-xylene regions of a Raman spectra. Therefore, the region of interest must be clearly stated in the region
    parameter.

    In general, the curve_fit function can be modified to suit the requirements of a different region of interest. The
    programmer must ensure that the residual is well-defined in the Residuals.py submodule and that the initial guesses
    in the parameters file is appropriate.

    The function can also be modified to return different fit parameters like the AUC, FWHM, height, etc.

    :param residuals: Function imported from Residuals module which is the objective function to be minimized.
    :param parameters: Parameter Object which contains all the relevant parameters for curve fitting.
    :param x: Numpy array of x-values
    :param y: Numpy array of y-values
    :param region: String indicating either 'vinyl' or 'pxylene'. This parameter is crucial because it will
                   trigger different fitting functions for calculating the AUC.

    :return: fit_params - Ordered dictionary of best fit parameters that can best fit the data
             r2score - Float of the calculated r2 score between fitted curve and actual data
             area - Float of the calculated AUC of the selected peak
    """
    mini = Minimizer(residuals, parameters, fcn_args=(x, y))  # Initialize Minimizer object
    out = mini.leastsq()  # Use Levenberg-Marquardt minimization to perform a fit.
    best_fit = y + out.residual
    # out.residual is a Numpy array of the minimized objective function when using the best-fit values
    # of the parameters. The best fit curve is therefore the y values plus the minimized residuals.
    # best_fit is also a Numpy array.

    fit_params = out.params.valuesdict()  # Returns an ordered dictionary of parameter values.
    r2score = r2_score(y, best_fit)  # Computes the r2 score between the best_fit and y.

    # Use a try-except clause to test whether region was set to 'vinyl' or 'pxylene'
    # If region was set to vinyl, use the lorentzian lineshape and the relevant peak parameters
    # If region was set to pxylene, use the split-lorentzian lineshape and the relevant peak parameters
    # If region was set to neither, raise an exception.
    try:
        if region == 'vinyl':
            y_fit = lorentzian(x, fit_params['p2amp'], fit_params['p2center'], fit_params['p2width'])
        elif region == 'pxylene':
            y_fit = split_lorentzian(x, fit_params['p3amp'], fit_params['p3center'], fit_params['p3width_left'],
                                     fit_params['p3width_right'])
        elif region != 'vinyl' or region != 'pxylene':
            raise Exception
    except Exception as e:
        print('Please specify in strings whether the region is vinyl or pxylene in curve_fit function')

    # Integrate the area below y_fit to get the AUC.
    area = integrate.simpson(y_fit, x)

    return fit_params, r2score, area
