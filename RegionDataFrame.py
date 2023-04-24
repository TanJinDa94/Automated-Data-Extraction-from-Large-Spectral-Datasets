import pandas as pd
import numpy as np


def find_nearest(array, value):
    """
    Find the nearest element in an array to the user defined value, as well the index of the element in the array.
    Methodology: Use numpy broadcasting and subtract the array by the value, to give a new difference array.
    Then use the .argmin() method to find the index of the lowest difference in the difference array.
    Original element can be easily extracted using array[index].

    :param array: Numpy array
    :param value: Integer or float
    :return: index - Index of element in the array
             element - Element of the index.
    """
    # Note that np.abs(array - value) is an array in itself, we can call it the difference array.
    # Note that the .argmin() method returns the indices of the minimum values along an axis.
    index = (np.abs(array - value)).argmin()
    element = array[index]

    return index, element


def region_df_slice(col_indices_filename, raw_data_filename):
    """
    Truncates dataframe by pandas index slicing, based on the indices provided by the user in the excel file.

    This function was specifically written for spitting the dataframe into vinyl and p-xylene regions but can be
    modified to suit the needs of the user.

    :param col_indices_filename: String containing the filename with extension of .xlsx containing
                                 column indices set by the user.
    :param raw_data_filename: String containing the filename with extension of .xlsx containing all extracted
                              Raman spectra

    :return: df_vinyl: DataFrame truncated to be between the column indices for vinyl region set by the user.
             df_pxylene: DataFrame truncated to be between the column indices for pxylene region set by the user.
    """
    df = pd.read_csv(raw_data_filename)  # Import raw Excel file

    # Import the excel file containing the column indices for slicing and convert the DataFrame into a dictionary
    # amenable for usage in index slicing.
    df_col_indices = pd.read_excel(col_indices_filename, header=None, index_col=0)
    d = df_col_indices.to_dict()[1]

    # Truncate DataFrame using pandas index slicing.
    df_vinyl = df.iloc[:, d['vinyl_left']:d['vinyl_right']]
    df_pxylene = df.iloc[:, d['pxylene_left']:d['pxylene_right']]

    return df_vinyl, df_pxylene

