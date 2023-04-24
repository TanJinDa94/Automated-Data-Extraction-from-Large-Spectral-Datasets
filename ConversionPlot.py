import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

conversion_df = pd.read_csv('df_conversion.csv')
error_df = pd.read_csv('df_error.csv')

plt.figure(figsize=(15, 10))

for index, rows in conversion_df.iterrows():
    x = np.array(conversion_df.columns[1:], dtype=float)  # Ignore Condition column during plotting

    y = rows[1:]  # Ignore Condition column during plotting

    plt.subplot(3, 3, (1 + index))

    plt.title('Condition ' + str(1 + index))
    # np.round up to 1 d.p. for slug identity, then convert to string.

    plt.xlabel('Time/ min')

    plt.ylabel('Conversion/ %')

    plt.xticks(x)  # Set xticks to relevant time intervals.

    plt.errorbar(x=x, y=y, yerr=error_df.iloc[index, 1:], capsize=5, fmt='bo')

plt.tight_layout()
plt.savefig('conversion_plots.png')
plt.show()

