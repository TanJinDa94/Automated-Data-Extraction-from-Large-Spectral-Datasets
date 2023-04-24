import matplotlib.pyplot as plt
import numpy as np


def simple_line_plot(x, y, x_label, y_label, title, save_name):
    """
    A simple function for graph plotting.
    x and y are arrays containing the x and y values, respectively.
    x and y must be of the same length for plotting purposes.

    :param x: List or numpy array for x values.
    :param y: List or numpy array for y values.
    :param x_label: String which labels the x-axis.
    :param y_label: String which labels the y-axis.
    :param title: String which represents the title.
    :param save_name: String which contains the file name with the file format.

    :return: None. Plots the graph.
    """
    plt.figure(figsize=(15, 8))

    plt.plot(x, y)
    plt.xlabel(x_label, fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel(y_label, fontsize=14)
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=14)

    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.show()


def subplot_2_by_1(x1, y1, x2, y2, x_label, y_label, title1, title2, save_name):
    """
    A simple function for plotting a graph side-by-side for comparison purposes. x1 and y1 are x and y-values for
    the first plot, while x2 and y2 are x and y-values for the second plot. The first plot is plotted on the left,
    while the second plot is plotted on the right.

    x1 and y1 are arrays containing the x and y values, respectively.
    x1 and y1 must be of the same length for plotting purposes.

    The same applies for x2 and y2.

    :param x1: List or numpy array for x values for the first plot.
    :param y1: List or numpy array for y values for the first plot.
    :param x2: List or numpy array for x values for the second plot.
    :param y2: List or numpy array for y values for the second plot.
    :param x_label: String which labels the x-axis.
    :param y_label: String which labels the y-axis.
    :param title1: String which represents the title for the first plot.
    :param title2: String which represents the title for the second plot.
    :param save_name: String which contains the file name with the file format.

    :return: None. Plots the graph.
    """
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    plt.plot(x1, y1)
    plt.xlabel(x_label, fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel(y_label, fontsize=14)
    plt.yticks(fontsize=12)
    plt.title(title1, fontsize=14)

    plt.subplot(1, 2, 2)
    plt.plot(x2, y2)
    plt.xlabel(x_label, fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel(y_label, fontsize=14)
    plt.yticks(fontsize=12)
    plt.title(title2, fontsize=14)

    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.show()


def baseline_subtraction_plot(x, y_original, y_baseline, y_subtracted, x_label, y_label, title1, title2, save_name):
    """
    A simple function for plotting a graph side-by-side for comparison purposes. x is the array containing x-values
    for the region of interest.

    The first plot contains the plot of the region of interest before baseline correction, and a straight line
    representing the baseline.

    The second plot contains the plot of the region of interest after baseline correction.

    :param x: List or numpy array for x values for the region of interest.
    :param y_original: List or numpy array for y values for the original y-values.
    :param y_baseline: List or numpy array for y values for the baseline.
    :param y_subtracted: List or numpy array for y values after baseline subtraction.
    :param x_label: String which labels the x-axis.
    :param y_label: String which labels the y-axis.
    :param title1: String which represents the title for the first plot.
    :param title2: String which represents the title for the second plot.
    :param save_name: String which contains the file name with the file format.

    :return: None. Plots the graph.
    """
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    plt.plot(x, y_original, label='Original Intensity')
    plt.plot(x, y_baseline, label='Baseline Intensity')
    plt.xlabel(x_label, fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel(y_label, fontsize=14)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)
    plt.title(title1, fontsize=14)

    plt.subplot(1, 2, 2)
    plt.plot(x, y_subtracted, label='Corrected Intensity')
    plt.plot(x, np.zeros(shape=len(x)), label='New Baseline')
    plt.xlabel(x_label, fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel(y_label, fontsize=14)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)
    plt.title(title2, fontsize=14)

    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.show()


def fitting_comparison(x, y, best_fit1, best_fit2, fit_params1, fit_params2,
                       x_label, y_label, title1, title2, save_name):
    """
    A function to plot fitted peaks side by side for a comparison.

    :param x: List or numpy array for x values for the region of interest.
    :param y: List or numpy array for y values for the region of interest.
    :param best_fit1: List or numpy array for best-fit y-values using the 1st functional form.
    :param best_fit2: List or numpy array for best-fit y-values using the 2nd functional form.
    :param fit_params1: Dictionary containing the fitted parameters using the 1st functional form.
    :param fit_params2: Dictionary containing the fitted parameters using the 2nd functional form.
    :param x_label: String which labels the x-axis.
    :param y_label: String which labels the y-axis.
    :param title1: String which represents the title for the first plot.
    :param title2: String which represents the title for the second plot.
    :param save_name: String which contains the file name with the file format.

    :return: None. Plots the graph.
    """
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    plt.title(title1, fontsize=14)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.plot(x, y, '#606060', linewidth=2, label='Original Peak')
    plt.plot(x, y, 'bo', markersize=1)
    plt.plot(x, best_fit1, 'r--', label='Best Fit $R^{2}$ = ' + str(fit_params1['r2_score'].round(decimals=3)),
             linewidth=2)
    plt.plot(x, best_fit1, '#32CD32', label='AUC of Best Fit')
    plt.fill_between(x, 0, best_fit1, facecolor='#32CD32', alpha=0.7)
    plt.legend(loc='best', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(1, 2, 2)
    plt.title(title2, fontsize=14)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.plot(x, y, '#606060', linewidth=2, label='Original Peak')
    plt.plot(x, y, 'bo', markersize=2)
    plt.plot(x, best_fit2, 'r--', label='Best Fit $R^{2}$ = ' + str(fit_params2['r2_score'].round(decimals=3)),
             linewidth=2)
    plt.plot(x, best_fit2, '#32CD32', label='AUC of Best Fit')
    plt.fill_between(x, 0, best_fit2, facecolor='#32CD32', alpha=0.9)
    plt.legend(loc='best', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.show()
