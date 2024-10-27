import pandas as pd
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler

dataset_dir = 'dataset'

# Load datasets
low_scrap = pd.read_csv(osp.join(dataset_dir, 'low_scrap.csv'))
high_scrap = pd.read_csv(osp.join(dataset_dir, 'high_scrap.csv'))

# Display basic info
print("Low Scrap Data:")
print(low_scrap.info())
print("\nHigh Scrap Data:")
print(high_scrap.info())

# Display first few rows to understand the structure
print(low_scrap.head())
print(high_scrap.head())

print(low_scrap.describe()['Station5_mp_85'])
print(high_scrap.describe()['Station5_mp_85'])

# low = low_scrap["Station5_mp_85"]
# high = high_scrap["Station5_mp_85"]

for column in low_scrap.columns:
    if "Station5" in column:
        low = low_scrap[column]
        high = high_scrap[column]
        plt.clf()
        plt.scatter(np.arange(low.values.shape[0]), np.sort(low), label='low')
        plt.scatter(np.arange(high.values.shape[0]), np.sort(high), label='high')
        plt.legend()
        plt.savefig(f'Claudio/Plots/{column}.png')
