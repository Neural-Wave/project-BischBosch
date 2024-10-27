
import pickle

import matplotlib.pyplot as plt
import networkx as nx

from data import load_dataset
from data import compute_naive_adj_mat, adjacency_to_digraph
# from data import plot_digraph

# from cdt.causality.graph import CGNN, GES, SAM

# from castle.algorithms import DAG_GNN
# from castle.metrics import MetricsDAG
from castle.common import GraphDAG

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc

from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

dataset = load_dataset(shuffle=True)
measurements = dataset.drop(columns='label').columns
npy_data = dataset.drop(columns='label').to_numpy()  # Load your data here
print(f"Data shape: {npy_data.shape}")

# adj_matrix = compute_naive_adj_mat(measurements)
# digraph = adjacency_to_digraph(adj_matrix, measurements)

# print("Adjacency Matrix:\n", adj_matrix)
# plot_digraph('naive_graph', digraph)

# model = CGNN(train_epochs=5, test_epochs=5)
# model = GES()
# causal_graph = model.orient_directed_graph(data=dataset.drop(columns='label'), dag=digraph)

# Initialize and train the DAG-GNN model
# model = DAG_GNN(device_type='gpu')
# model.learn(npy_data, columns=measurements)
# causal_graph = model.causal_matrix
# print("Inferred Causal Matrix:\n", causal_graph)
# plot_digraph('causal_graph', causal_graph)


bk = BackgroundKnowledge()
for m in measurements:
    bk = bk.add_node_to_tier(GraphNode(m), int(m.split('_')[0][-1]))

# causal_graph, edges = fci(npy_data, background_knowledge=bk)
causal_graph = pc(npy_data, background_knowledge=bk, mvpc=True)

# model = SAM(train_epochs=10, test_epochs=10, batch_size=8, nruns=1)
# causal_graph = model.predict(dataset.drop(columns='label').astype('float32'), graph=digraph)

with open('causal_graph.pkl', 'wb') as f:
    pickle.dump(causal_graph, f)

causal_graph.to_nx_graph()
GraphDAG(est_dag=nx.adjacency_matrix(causal_graph.nx_graph).toarray().T)
plt.savefig('causal_graph_adj.png')

pdy = GraphUtils.to_pydot(causal_graph.G, labels=measurements)
pdy.write_png('causal_graph.png')