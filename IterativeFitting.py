import numpy as np
from BaselineSubtractionFunction import baseline_subtraction_function
from Parameters import define_region_parameters
from CurveFitting import curve_fit


def iterative_fitting(df_region, parameter_filename, region, residuals):
    """
    Iterate through every row of the region of interest and execute the curve fitting.

    :param df_region: pandas DataFrame already truncated to contain the region of interest
    :param parameter_filename: String of filename with file extension
    :param region: String indicating the region of interest
    :param residuals: Function which acts as the objective function to be minimised.

    :return: bestfit_params_list - List of Ordered Dictionary of Best fit parameters that can best fit the curve
             r2_score_list - List of R2 scores of the fit
             area_list - List of AUC of Vinyl Peak
    """
    bestfit_params_list = []  # List of Ordered Dictionary of Best fit parameters that can best fit the curve
    r2_score_list = []  # List of R2 scores of the fit
    area_list = []  # List of AUC of Vinyl Peak

    parameters = define_region_parameters(parameter_filename)

    for index, series in df_region.iterrows():  # Iterate over DataFrame rows as (index, Series) pairs.

        x = np.array(series.index, dtype=float)
        y = series

        linear_fit, y_subtracted = baseline_subtraction_function(region=y)

        bestfit_params, r2score, area = curve_fit(residuals=residuals,
                                                  parameters=parameters,
                                                  x=x,
                                                  y=y_subtracted,
                                                  region=region)

        bestfit_params_list.append(bestfit_params)
        r2_score_list.append(r2score)
        area_list.append(area)

    return bestfit_params_list, r2_score_list, area_list
