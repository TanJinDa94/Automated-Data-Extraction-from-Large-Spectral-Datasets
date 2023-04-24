from IterativeFitting import iterative_fitting
from Residuals import residuals_vinyl, residuals_pxylene
from Consolidate import aggregate_ratio
from RegionDataFrame import region_df_slice
import pandas as pd

vinyl_parameter = 'vinyl_parameters.xlsx'
pxylene_parameter = 'pxylene_parameters.xlsx'
column_indices = 'column_indices.xlsx'

file_list = ['df_t0.csv', 'df_t0_repeat.csv', 'df_t30.csv', 'df_t60.csv', 'df_t90.csv', 'df_t120.csv']
file_number = 0

for file in file_list:
    file_number += 1
    print('Currently Processing File Number ' + str(file_number) + ' out of ' + str(len(file_list)))
    print('File name is: ', file)

    df = pd.read_csv(file)

    df_vinyl, df_pxylene = region_df_slice(column_indices, file)

    vinyl_bestfit_params, vinyl_r2_score, vinyl_area = iterative_fitting(df_region=df_vinyl,
                                                                         parameter_filename=vinyl_parameter,
                                                                         region='vinyl',
                                                                         residuals=residuals_vinyl)

    pxylene_bestfit_params, pxylene_r2_score, pxylene_area = iterative_fitting(df_region=df_pxylene,
                                                                               parameter_filename=pxylene_parameter,
                                                                               region='pxylene',
                                                                               residuals=residuals_pxylene)

    df_ratio = aggregate_ratio(df=df,
                               vinyl_area=vinyl_area, vinyl_r2_score=vinyl_r2_score,
                               pxylene_area=pxylene_area, pxylene_r2_score=pxylene_r2_score,
                               filename=file[:-4])


print('Finished Processing all Files.')
