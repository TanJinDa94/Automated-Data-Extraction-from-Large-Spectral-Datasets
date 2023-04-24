import pandas as pd
import numpy as np
from lmfit import Parameters


def define_region_parameters(filename):
    """
    Reads an Excel file and imports the user defined parameter values and bounds within the excel file
    into a Parameter object, amenable for usage during curve fitting using the lmfit library.

    :param filename: String containing the excel filename with the file extension
    :return: parameters - Parameters object which contains all relevant use defined parameters
    """
    df = pd.read_excel(filename)  # Read excel file with parameters filled in.
    df = df.replace(np.nan, None)  # Replace empty cells with None.

    parameters = Parameters()  # Instantiate Parameters object

    # Iterate through the rows of the DataFrame and add defined parameters to parameters object.
    for index, series in df.iterrows():
        parameters.add(*series.values)

    return parameters
