import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import make_dot
import matplotlib.pyplot as plt
from castle.common import GraphDAG

np.set_printoptions(precision=3, suppress=True)
np.random.seed(100)

def get_data(with_period=False):
    low = pd.read_csv('dataset/low_scrap.csv')
    high = pd.read_csv('dataset/high_scrap.csv')
    if with_period:
        low['period'] = 0
        high['period'] = 1
    return pd.concat([low, high], ignore_index=True)

dataset = get_data()

model = lingam.DirectLiNGAM()
model.fit(dataset)


pred_dag_expert = model._adjacency_matrix
# 
# Compare the predicted graph vs the ground truth 
GraphDAG(
    est_dag=pred_dag_expert
)
plt.axvline(85., alpha=0.1)
uint8_matrix = np.array(pred_dag_expert, dtype=np.uint8)

np.savetxt(f'Claudio/DAG_matrix_Lingam.txt', uint8_matrix, fmt='%d', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
plt.savefig(f'Claudio/DAG_matrix_Lingam.jpg')



# make_dot(model.adjacency_matrix_, labels=np.array(dataset.columns))

