from castle.algorithms.lingam.direct_lingam import DirectLiNGAM
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd 

def get_data(with_period=False):
    low = pd.read_csv('dataset/low_scrap.csv')
    high = pd.read_csv('dataset/high_scrap.csv')
    if with_period:
        low['period'] = 0
        high['period'] = 1
    return pd.concat([low, high], ignore_index=True)

dataset = get_data()

prior = -np.ones((98, 98))

gt_15_rows = np.loadtxt('dataset/gt_15_rows.txt')
gt_lower = np.loadtxt('dataset/lower_triangular_gt.txt')

for i in range(prior.shape[0]):
    for j in range(i+1):
        prior[i, j] = gt_lower[i, j]

prior[:15] = gt_15_rows

model = DirectLiNGAM(prior_knowledge=gt_lower)

model.learn(dataset.values, columns=dataset.columns)

causal_matrix = model.causal_matrix

np.savetxt(f'Claudio/DAG_matrix_Direct_Lingam.txt', causal_matrix, fmt='%d', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
plt.savefig(f'Claudio/DAG_matrix_Direct_Lingam.jpg')


