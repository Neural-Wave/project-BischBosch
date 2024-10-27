import numpy as np
import matplotlib.pyplot as plt

GT = np.loadtxt('DAG_MATRIX.txt')
plt.imshow(np.logical_not(GT), cmap='gray')
plt.savefig('ciuao.png')