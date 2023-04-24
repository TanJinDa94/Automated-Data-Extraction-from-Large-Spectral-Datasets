import numpy as np
import pandas as pd
from uncertainties import ufloat

# Create list of filenames to be read. Ensure that the files are listed in the correct order!
filenames = ['df_t0_ratio.csv', 'df_t0_repeat_ratio.csv', 'df_t30_ratio.csv',
             'df_t60_ratio.csv', 'df_t90_ratio.csv', 'df_t120_ratio.csv']

dataframes = [pd.read_csv(file) for file in filenames]  # # Use list comprehension to generate list of read dataframes


def create_mean_df(dataframes):
    """
    Create a dataframe to store the mean AUC ratio values of each condition across residence times.

    :param dataframes: List of DataFrames containing the mean AUC ratio of conditions in the different residence times.

    :return: df_mean: DataFrame containing the mean ratio of the AUCs associated with each condition across
                      residence times.
    """
    # Create an initial dataframe condition and mean columns of the t0 dataframe
    df_mean = dataframes[0][['condition', 'mean']]

    # Concatenate subsequent dataframe's mean columns along the columns axis and ignore the index.
    for file in dataframes[1:]:
        df_mean = pd.concat([df_mean, file['mean']], axis=1, ignore_index=True)

    # Define the column headers.
    col_names = {0: 'Condition',
                 1: 't0_mean',
                 2: 't0_mean_repeat',
                 3: 't30_mean',
                 4: 't60_mean',
                 5: 't90_mean',
                 6: 't120_mean'}
    # Rename columns
    df_mean.rename(columns=col_names, inplace=True)

    return df_mean


def create_std_df(dataframes):
    """
    Create a dataframe to store the standard deviation of the AUC ratio values of each condition across residence times.

    :param dataframes: List of DataFrames containing the standard deviation of the AUC ratio of conditions in the
                       different residence times.

    :return: df_std: DataFrame containing the standard deviation of the AUC ratios associated with each condition across
                     residence times.
    """
    # Create an initial dataframe condition and std columns of the t0 dataframe
    df_std = dataframes[0][['condition', 'std']]

    # Concatenate subsequent dataframe's std columns along the columns axis and ignore the index.
    for file in dataframes[1:]:
        df_std = pd.concat([df_std, file['std']], axis=1, ignore_index=True)

    # Define the column headers.
    col_names = {0: 'Condition',
                 1: 't0_std',
                 2: 't0_std_repeat',
                 3: 't30_std',
                 4: 't60_std',
                 5: 't90_std',
                 6: 't120_std'}
    # Rename columns
    df_std.rename(columns=col_names, inplace=True)

    return df_std


def create_ufloat_df(df_mean, df_std):
    """
    A function that takes in both df_mean and df_std and creates a single df_ufloat containing the mean with
    standard deviation as a single object in each DataFrame entry. In this ufloat format, conversion calculation and
    error propagation would be immensely simpler.

    df_mean, df_std and df_ufloat are of the same shape.

    :param df_mean: DataFrame containing the mean AUC ratios.
    :param df_std: DataFrame containing the standard deviation of the AUC ratios.

    :return: df_ufloat - DataFrame containing the mean with standard deviation as a single object
                         in each DataFrame entry.
    """
    for file in filenames:  # Iterate through each file in their stated order.
        l1 = []  # Create a list called l1 to store lists, eventually becoming a list of lists
        for column_index in range(1, 7):  # Iterate through each column, ignoring the 0th column for condition
            l2 = []  # Create a temporary list l2, to store mean with std object
            for row_index in range(0, 9):  # Iterate through each row
                mean_with_std = ufloat(df_mean.iloc[row_index, column_index], df_std.iloc[row_index, column_index])
                # Combine mean and std into a single object using the ufloat function
                l2.append(mean_with_std)
                # Append mean_with_std into l2 list, and do this for each row
            l1.append(l2)  # Append l2 list into l1 list, and do this for each column

    array = np.array(l1).T  # Convert to numpy array and transpose
    columns = [0, '0_repeat', 30, 60, 90, 120]  # Create the column headers
    df_ufloat = pd.DataFrame(array, columns=columns)  # Create the DataFrame
    df_ufloat.insert(loc=0, column='Condition', value=df_mean.Condition.values)  # Add back in the conditions column
    return df_ufloat


def calc_conv_and_error(df_ufloat):
    """
    Calculate the conversion and propagate the error leveraging upon the ufloat package.
    pandas broadcasting is used for conversion calculation and error propagation.

    :param df_ufloat:  DataFrame containing the mean and std object in each entry.

    :return:  df_conv_and_error - DataFrame containing the conversion and propagated error object in each entry.
    """
    # Create a DataFrame skeleton first and then define the columns as the Series values are calculated.
    df_conv_and_error = pd.DataFrame()
    df_conv_and_error = pd.concat([df_conv_and_error, df_ufloat['Condition']], axis=1)

    # Pre-define the t0 and t0_repeat pandas Series, as they will be used as the benchmark for conversion calculations.
    # t0, t30, t60 and t90 will use t0 as the benchmark
    # t120 will use t0_repeat for the benchmark
    t0 = df_ufloat.iloc[:, 1]
    t0_repeat = df_ufloat.iloc[:, 2]

    df_conv_and_error[0] = ((1 - (df_ufloat.iloc[:, 1] / t0)) * 100)  # Calculate t0 conversion. Should be 0+/-0.

    i = 3  # Calculate the t30, t60 and t90 conversions. Start from column index 3.
    for time in [30, 60, 90]:
        df_conv_and_error[time] = ((1 - (df_ufloat.iloc[:, i] / t0)) * 100)
        i += 1

    df_conv_and_error[120] = ((1 - (df_ufloat.iloc[:, -1] / t0_repeat)) * 100)  # Calculate t120 conversion.

    return df_conv_and_error


def conversion_and_error(df_conv_and_error):
    """
    Create separate conversion and error DataFrames from df_conv_and_error for plotting purposes.

    :param df_conv_and_error: DataFrame containing the conversion and propagated error object in each entry.

    :return: conversion_df - DataFrame containing only conversion floats.
             error_df - DataFrame containing only error floats.
    """
    conversion = []
    error = []
    for index, column in df_conv_and_error.iloc[:, 1:].items():  # df.items() iterate via columns
        conversion_temp = []
        error_temp = []

        for item in column.values:
            conversion_temp.append(item.nominal_value)
            error_temp.append(item.std_dev)

        conversion.append(conversion_temp)
        error.append(error_temp)

    # Create separate dataframs for conversion_df and error_df
    columns = [0, 30, 60, 90, 120]
    conversion_df = pd.DataFrame(np.array(conversion).T, columns=columns)
    conversion_df.insert(loc=0, column='Condition', value=df_mean.Condition.values)
    error_df = pd.DataFrame(np.array(error).T, columns=columns)
    error_df.insert(loc=0, column='Condition', value=df_mean.Condition.values)

    # Save df to csv files
    conversion_df.to_csv('df_conversion.csv', index=False)
    error_df.to_csv('df_error.csv', index=False)
    return conversion_df, error_df


df_mean = create_mean_df(dataframes)
df_std = create_std_df(dataframes)
df_ufloat = create_ufloat_df(df_mean, df_std)
df_conv_and_error = calc_conv_and_error(df_ufloat)
conversion_df, error_df = conversion_and_error(df_conv_and_error)

print('df_conversion.csv and df_error.csv are saved.')
