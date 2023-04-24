import pandas as pd


def aggregate_ratio(df, vinyl_area, vinyl_r2_score, pxylene_area, pxylene_r2_score, filename):
    """
    A function which calculates the aggregate mean AUC ratio and the standard deviation of the AUC ratio of multiple
    Raman spectra associated to their respective conditions.

    :param df: DataFrame of the excel file containing the original
    :param vinyl_area: List of all vinyl peak AUC.
    :param vinyl_r2_score: List of all vinyl region R2 scores.
    :param pxylene_area: List of all pxylene peak AUC.
    :param pxylene_r2_score: List of all pxylene region R2 scores.
    :param filename: String of the filename WITHOUT the extension.

    :return: df_ratio - DataFrame consisting only of the condition label, the mean ratio and the standard deviation of the ratio.
    """
    # Extract original index and condition labels from the raw DataFrame into an array of values
    original_index = df.iloc[:, 0].values
    condition = df.iloc[:, 1].values

    # Create a Dataframe using the appropriate column headers and lists of AUC and R2 scores.
    # The DataFrame should now contain the following information in each row:
    # The original index and condition for the spectra, the vinyl peak AUC and R2 score
    # and the pxylene peak AUC and R2 score. The number of rows should correspond to the total number of extracted
    # spectra in the original Excel file.
    d = {'Original Index': original_index,
         'Condition': condition,
         'Vinyl Peak AUC': vinyl_area,
         'Vinyl R2 Score': vinyl_r2_score,
         'p-xylene Peak AUC': pxylene_area,
         'p-xylene R2 Score': pxylene_r2_score
         }
    df_area = pd.DataFrame(d)

    # Filter out poorly fitted spectra. Only spectra with R2 score of fit for both vinyl and pxylene regions
    # above 0.95 should be kept for further calculations.
    # Create a column for the ratio between the vinyl peak AUC and the pxylene peak AUC.
    # Use the pandas .groupby() operation to cluster spectra of the same condition together, and use the .describe()
    # method to obtain summary statistics of the ratios.
    # The DataFrame arising from this operation has hierarchical indexes.
    # The total number of rows in this DataFrame should now be equal to the total number of conditions.
    df_area = df_area[(df_area['Vinyl R2 Score'] > 0.95) & (df_area['p-xylene R2 Score'] > 0.95)]
    df_area['Vinyl Divide p-xylene'] = df_area['Vinyl Peak AUC'] / df_area['p-xylene Peak AUC']
    df_area_stats = df_area[['Condition', 'Vinyl Divide p-xylene']].groupby('Condition').describe()

    # Instantiate a list of conditions to contain the condition labels without repeats, sorted in ascending order.
    condition = list(set(condition))  # Remove repeats using the set object, then convert it to list.
    condition.sort()  # Modifies list in place.

    # Extract the mean and standard deviation of the ratios using pandas multi-indexing.
    # The mean and standard deviation of the ratios will be used for further conversion and error calculations.
    mean = df_area_stats.loc[:, ('Vinyl Divide p-xylene', 'mean')].values
    std = df_area_stats.loc[:, ('Vinyl Divide p-xylene', 'std')].values

    # Create a DataFrame consisting only of the condition label, the mean ratio and the standard deviation of the ratio.
    df_ratio = pd.DataFrame({
        'condition': condition,
        'mean': mean,
        'std': std
    })

    df_ratio.to_csv(filename + '_ratio.csv', index=False)  # Write the DataFrame to a .csv file.

    return df_ratio
