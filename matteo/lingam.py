
import pickle

from data import load_dataset, compute_naive_adj_mat

import numpy as np
import matplotlib.pyplot as plt

from castle.common import GraphDAG
from castle.algorithms import DirectLiNGAM


dataset = load_dataset(shuffle=True)
measurements = dataset.drop(columns='label').columns
npy_data = dataset.drop(columns='label').to_numpy()
print(f"Data shape: {npy_data.shape}")

expert_knowledge = compute_naive_adj_mat(measurements, load_gt=True)

# Compare the predicted graph vs the ground truth 
GraphDAG(est_dag=expert_knowledge)
plt.savefig('matteo/expert_lingam.jpg')

g = DirectLiNGAM(prior_knowledge=expert_knowledge)
g.learn(npy_data)

with open('matteo/lingam_graph.pkl', 'wb') as f:
    pickle.dump(g.causal_matrix, f)

# Compare the predicted graph vs the ground truth 
GraphDAG(est_dag=g.causal_matrix.T)
plt.savefig('matteo/lingam_graph_adj.jpg')

uint8_matrix = np.array(g.causal_matrix, dtype=np.uint8)
# np.savetxt('matteo/DAG_matrix_with_gt_alpha_010.txt', uint8_matrix, fmt='%d', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)

