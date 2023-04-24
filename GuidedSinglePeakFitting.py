from Plotting import simple_line_plot, subplot_2_by_1, baseline_subtraction_plot, \
    fitting_comparison
from RegionDataFrame import find_nearest
from BaselineSubtractionFunction import baseline_subtraction_function
from Residuals import residuals_lorentzian, residuals_gaussian
from CurveFitting import lorentzian_curve_fit, gaussian_curve_fit
from lmfit import Parameters, Minimizer
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

error_comment = 'Please refer to the documentation and ensure that your data is ' \
                'structured properly before running this package.'

print('\nHi, welcome to the automated peak fitting package.')
print('This package automates the following processes: baseline subtraction, peak fitting and relevant spectra detail '
      'extraction such as AUC, FWHM, peak height or peak location\n')
print('\n==================================================\n')

prompt1 = str(input(
    'Is your data in the form of x-values (wavenumbers or wavelengths) as the column names, and y-values'
    ' (intensity, or absorption) as the values of a table, with each row representing a single spectrum? [y/n]\n'))

if prompt1 == 'y':
    print('\nUser asserts that data is structured properly. Proceed to ask user to input file name of dataset.')
    print('\n==================================================\n')

else:
    print(error_comment)
    sys.exit()

prompt2 = str(input('Please fill in the file name (including extension) of the .csv file that contains your data: '))

try:
    df = pd.read_csv(prompt2)
    x = np.array(df.columns, dtype=float)
    y = np.array(df.iloc[0, :], dtype=float)
    title = 'Plot of first spectrum (first row) in dataset'
    x_axis_label = str(input('\nWhat is the name of the x-axis and its units in brackets?'))
    y_axis_label = str(input('\nWhat is the name of the y-axis and its units in brackets?'))
    print('\nPlotting the first spectrum from the dataset.'
          '\nPlease take note of the leftmost and rightmost x-axis values which contains the region of'
          ' interest in the plot.')
    simple_line_plot(x, y, x_axis_label, y_axis_label, title, save_name='First Spectra Plot.png')
except:
    print('Unable to find file name. Please enter the correct file name.')
    sys.exit()

prompt3 = str(input('\nThe 1st spectrum from your dataset was plotted. Does the plot look correct? [y/n] \n'))

if prompt3 == 'y':
    print('\nUser asserts that the plot of the 1st spectrum looks correct.')
    print('\nProceed to define region of interest by the leftmost and rightmost x-values.')
    print('\n==================================================\n')
else:
    print(error_comment)
    sys.exit()

print('\nThe next step involves truncating the dataset to focus only on the region of interest. To do this, the script '
      'requires the left-most and right-most x-axis values of the region of interest.')

try:
    leftmost = int(input('\nWhat is the left-most x-axis value for the region of interest? '))
    leftmost_index, leftmost_element = find_nearest(x, leftmost)
    print('\nThe left-most x-value closest to ' + str(leftmost) + ' is ' + str(leftmost_element) +
          ' in column number ' + str(leftmost_index) + ' of the dataframe.')
    print('Column Number ' + str(leftmost_index) + ' will be used for dataframe slicing.')

    rightmost = int(input('\nWhat is the right-most x-axis value for the region of interest? '))
    rightmost_index, rightmost_element = find_nearest(x, rightmost)
    print('\nThe right-most x-value closest to ' + str(rightmost) + ' is ' + str(rightmost_element) +
          ' in column number ' + str(rightmost_index) + ' of the dataframe.')
    print('Column Number ' + str(rightmost_index) + ' will be used for dataframe slicing.')

except:
    print('Unable to find the appropriate x-values. Please enter the correct x-values.')
    sys.exit()

print('\nProceed to plot region of interest as defined by the leftmost and rightmost x-values input by user.')

region = df.iloc[:, leftmost_index: rightmost_index]
region_x = np.array(region.columns, dtype=float)
region_y = np.array(region.iloc[0, :], dtype=float)
title1 = 'Original plot prior to slicing'
title2 = 'Plot of region'
subplot_2_by_1(x1=x, y1=y, x2=region_x, y2=region_y, x_label=x_axis_label,
               y_label=y_axis_label, title1=title1, title2=title2,
               save_name='Plots before and after slicing.png')

prompt4 = str(input('\nThe plots before and after dataframe slicing were plotted.'
                    ' Do the plots look correct? [y/n] \n'))

if prompt4 == 'y':
    print('\nUser asserts that the plots looks correct. Proceed to prompt for baseline correction.')
    print('\n==================================================\n')
else:
    print('Please input the correct left-most and right-most x-values for the region of interest')
    sys.exit()

prompt5 = str(input('\nDo you need to do baseline subtraction for the region of interest? [y/n] \n'))

if prompt5 == 'y':
    print('\nUser requests for baseline subtraction. Proceed to conduct baseline correction.')
    linear_fit, y_subtracted = baseline_subtraction_function(region.iloc[0, :])
    title3 = 'Region of interest prior to baseline subtraction'
    title4 = 'Region of interest after baseline subtraction'
    baseline_subtraction_plot(x=region_x, y_original=region_y, y_baseline=linear_fit,
                              y_subtracted=y_subtracted, x_label=x_axis_label,
                              y_label=y_axis_label, title1=title3, title2=title4,
                              save_name='Plots before and after baseline subtraction.png')
    print('\n==================================================\n')

    prompt6 = str(input('\nThe spectra before and after baseline correction are plotted.'
                        ' Do the plots look correct? [y/n] \n'))
    if prompt6 == 'y':
        print('\nUser asserts that the plot after baseline correction is correct. '
              'Proceed to prompt for peak fitting.')
        print('\n==================================================\n')
    else:
        print('Please input the correct left-most and right-most x-values for the region of interest')
        sys.exit()
else:
    print('\nUser does not require baseline subtraction. Proceed to peak fitting directly.')
    print('\n==================================================\n')

print('\nAttempting to fit the peak using Lorentzian and Gaussian functional forms. Both functional forms require '
      '3 initial guesses to begin.\nYou will be prompted for intial guesses for 1) the center of the peak, '
      '2) the amplitude of the peak and 3) the half-width at half-maximum of the peak.\nOne can suggest better initial '
      'guesses after the first round of peak fitting. Good initial guesses can prevent non-convergence and speed up '
      'peak fitting.')

p1_center = int(input('\nThe center of the peak is the x-value for which the y-value of the peak is maximum. '
                      '\nMake an initial educated guess for the center of the peak on the x-axis: '))
p1_half_width = int(input('\nThe half-width of the peak is half of the full-width half-maximum (FWHM). '
                          '\nThe FWHM is the width of the peak at 50% of the maximum y-value of the peak.'
                          '\nMake an initial educated guess for the width of the peak: '))
p1_amplitude = int(
    input('\nThe amplitude of the peak is the scalar that multiplies the functional form expression such that '
          'the lineshape is unit-normalized.'
          '\nIn other words, the Area Under the Curve (AUC) of the lineshape is normalized to be equal to 1.'
          '\nAccount for the scales of both the x and y axes, to determine an appropriate guess for the '
          'numerical value for the amplitude. '
          '\nMake an initial educated guess for the amplitude of the peak: '))

print('\nProceeding to fit curves and plot the Lorentzian and Gaussian fits for comparison')
print('\n==================================================\n')

parameters = Parameters()
parameters.add(name='p1_amplitude', value=p1_amplitude, min=0)
parameters.add(name='p1_center', value=p1_center, min=0)
parameters.add(name='p1_half_width', value=p1_half_width, min=0)

best_fit1, fit_params1 = lorentzian_curve_fit(residuals_lorentzian, parameters, region_x, region_y)
best_fit2, fit_params2 = gaussian_curve_fit(residuals_gaussian, parameters, region_x, region_y)

curve_fitting_function = {0: lorentzian_curve_fit,
                          1: gaussian_curve_fit}

objective_function = {0: residuals_lorentzian,
                      1: residuals_gaussian}

fitting_comparison(region_x, region_y, best_fit1, best_fit2, fit_params1, fit_params2,
                   x_axis_label, y_axis_label, 'Lorentzian Fit', 'Gaussian Fit', 'Fitting Comparison Plot.png')

print('\nPeak fitted with both Lorentzian and Gaussian functional forms.')

prompt7 = str(input('\nDo the peak fittings look correct? [y/n]'))

if prompt7 == 'y':
    print('\nUser asserts that the peak fittings look correct.'
          '\nProceed to ask user about the preferred lineshape for peak fitting.')

elif prompt7 == 'n':
    print('\nUser is not satisfied with peak fittings.')
    print('\nPlease restart the script and ensure that instructions are followed carefully.')
    sys.exit()

prompt8 = int(input('\nWhich lineshape fits the peak the best? A higher R squared value indicates a better fit.'
                    '\nIf the lorentzian function form is a better fit, please key in 0.'
                    '\nIf the gaussian function form is a better fit, please key in 1.\n'))

if prompt8 == 0:
    print('\nUser has chosen the Lorentzian functional form.')

elif prompt8 == 1:
    print('\nUser has chosen the Gaussian functional form.')

print('\n==================================================\n')

prompt9 = str(input('\nProceed to do peak fitting for all spectra in the dataset? [y/n]'))

if prompt9 == 'y' and prompt5 == 'n':
    print('\nPeak fitting for all spectra without baseline subtraction commencing.')
    results = []
    for index, row in region.iterrows():
        print('Currently Fitting Spectra Number ' + str(index) + ' out of ' + str(len(region)))
        y = np.array(row.values, dtype=float)

        best_fit, fit_params = curve_fitting_function[prompt8](objective_function[prompt8], parameters,
                                                               region_x, y)
        results.append(fit_params)

    print('Peak fitting has ended.'
          '\nFitting results for all spectra will be saved in a .csv file.'
          '\nSummary statistics of the fitting results will be saved in a .csv file.')

    results = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)  # Print full dataframe without truncation.

    results.to_csv(prompt2[:-4] + '_results.csv')
    results.describe().to_csv(prompt2[:-4] + '_summary.csv')

elif prompt9 == 'y' and prompt5 == 'y':
    print('\nPeak fitting for all spectra with baseline subtraction commencing.')
    results = []
    for index, row in region.iterrows():
        print('Currently Fitting Spectra Number ' + str(index) + ' out of ' + str(len(region)))
        y = np.array(row.values, dtype=float)
        linear_fit, y_subtracted = baseline_subtraction_function(region.iloc[index, :])

        best_fit, fit_params = curve_fitting_function[prompt8](objective_function[prompt8], parameters,
                                                               region_x, y_subtracted)
        results.append(fit_params)

    print('Peak fitting has ended.'
          '\nFitting results for all spectra will be saved in a .csv file.'
          '\nSummary statistics of the fitting results will be saved in a .csv file.')

    results = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)  # Print full dataframe without truncation.

    results.to_csv(prompt2[:-4] + '_results.csv')
    results.describe().to_csv(prompt2[:-4] + '_summary.csv')

elif prompt9 == 'n':
    print('User does not want to continue with peak fitting for all spectra.')
    print('End of script')
    print('####################################################################')
    sys.exit()

else:
    print('Invalid option. Please run the script again.')
    sys.exit()

prompt10 = str(input('\nProceed to plot fitted parameters against spectra index? [y/n]'))

if prompt10 == 'y':

    print('\nPlotting fitted parameters against spectra index.\n')

    spectra_index = np.array(results.index, dtype=int)
    fig, axs = plt.subplots(2, len(results.columns), figsize=(15, 10))
    for index, column in enumerate(results.columns):
        y_values = results[column].values

        axs[0, index].scatter(x=spectra_index, y=y_values, s=1)
        axs[0, index].set_title(str(column) + ' Scatter Plot')
        axs[0, index].set_xlabel('Spectra Index')

        axs[1, index].boxplot(results[column].values)
        axs[1, index].set_title(str(column) + ' Box Plot')
        axs[1, index].set_xlabel(str(column))

    plt.tight_layout()
    plt.savefig(prompt2[:-4] + '_summary_plot.png')
    plt.show()

    print('Summary plot will be saved in a .csv file.\n')

elif prompt10 == 'n':
    print('User does not want to plot fitted parameters against spectra index.')
    print('End of script')
    print('####################################################################')
    sys.exit()

else:
    print('Invalid option. Please run the script again.')
    sys.exit()

print('####################################################################')
print('#########################END OF SCRIPT##############################')
print('####################################################################')

sys.exit()
