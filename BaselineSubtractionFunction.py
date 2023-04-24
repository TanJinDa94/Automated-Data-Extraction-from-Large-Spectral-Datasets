import numpy as np
from numpy.polynomial import Polynomial


def baseline_subtraction_function(region):
    """
    Use linear least-squares to fit a linear baseline to the left-most 5 and right-most 5 x and y values of the region
    using the Polynomial module from the NumPy Library. The output of the fitting gives the coefficients a and b
    of the linear equation y = ax + b.

    From the fitting coefficients a and b, construct linear_fit, which is a NumPy array with the same length as array y.
    This linear_fit array is the baseline, which contains all intensities of the baseline.

    Subtract the baseline array linear_fit from the original y-values to obtain a baseline-corrected array named
    y_subtracted.

    :param region: Pandas series of defined region of interest, extracted from dataset containing extracted spectra.
    :return: linear_fit - Numpy array containing the linear fit of the left-most 5 and right-most 5 x and y values,
                          representing the intensities of the baseline.
             y_subtracted - Numpy array containing the y-values after baseline subtraction
    """
    # Convert wavenumber labels and intensity values from series to numpy arrays of datatype float for easy manipulation
    x = np.array(region.index, dtype=float)
    y = np.array(region.values, dtype=float)

    # Concatenate leftmost 5 and rightmost 5 x and y values into separate single np arrays
    x_extreme = np.concatenate((x[:5], x[-5:]))
    y_extreme = np.concatenate((y[:5], y[-5:]))

    # Fit a linear line (baseline) through the scatter-plot defined by x_extreme and y_extreme np arrays.
    # Extract the slope and intercept from the fitted linear baseline.
    # Reconstruct the fitted linear baseline in the form of an array with the same number of elements as array y.
    output = Polynomial.fit(x=x_extreme,
                            y=y_extreme,
                            deg=1)
    coefficients = output.convert().coef
    linear_fit = coefficients[1] * x + coefficients[0]

    y_subtracted = y - linear_fit  # Subtract linear_fit array from y array.

    return linear_fit, y_subtracted
