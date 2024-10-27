
import pickle

from data import load_dataset

import networkx as nx
import matplotlib.pyplot as plt

from castle.common import GraphDAG
from castle.algorithms import GES


dataset = load_dataset(shuffle=True)
measurements = dataset.drop(columns='label').columns
npy_data = dataset.drop(columns='label').to_numpy()
print(f"Data shape: {npy_data.shape}")

g = GES()
g.learn(npy_data)

with open('matteo/ges_graph.pkl', 'wb') as f:
    pickle.dump(g.causal_matrix, f)

# causal_graph.to_nx_graph()
GraphDAG(est_dag=g.causal_matrix.T)
plt.savefig('matteo/ges_graph_adj.png')

# pdy = GraphUtils.to_pydot(g.causal_matrix, labels=measurements)
# pdy.write_png('ges_graph.png')