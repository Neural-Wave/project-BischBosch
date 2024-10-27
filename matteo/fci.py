
import pickle

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from data import load_dataset, load_npy_txt_from_file

from castle.common import GraphDAG

from causallearn.search.ConstraintBased.FCI import fci

from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

dataset = load_dataset(shuffle=True)
measurements = dataset.drop(columns='label').columns
npy_data = dataset.drop(columns='label').to_numpy()
print(f"Data shape: {npy_data.shape}")

nodes = []
for m in measurements:
    nodes.append(GraphNode(f"X{int(m.split('_')[-1])+1}"))

bk = BackgroundKnowledge()
for i, src_station in enumerate(measurements):
    for j, dst_station in enumerate(measurements):
        if int(src_station.split('_')[0][-1]) > int(dst_station.split('_')[0][-1]):
            bk = bk.add_forbidden_by_node(nodes[i], nodes[j])

gt = load_npy_txt_from_file('dataset/gt_15_rows.txt')
for i in range(gt.shape[0]):
    for j in range(gt.shape[1]):
        edge = gt[i,j]
        if edge == 0:
            bk = bk.add_forbidden_by_node(nodes[i], nodes[j])
        else:
            bk = bk.add_required_by_node(nodes[i], nodes[j])

causal_graph, edges = fci(npy_data, background_knowledge=bk)

with open('matteo/fci_graph.pkl', 'wb') as f:
    pickle.dump(causal_graph, f)

# with open('matteo/fci_graph.pkl', 'rb') as f:
#     causal_graph = pickle.load(f)

# Get list of nodes and create a mapping to indices
nodes = causal_graph.get_nodes()
node_index_map = {node: idx for idx, node in enumerate(nodes)}

# Initialize adjacency matrix
num_nodes = len(nodes)
adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

# Fill the adjacency matrix based on edge information
for edge in causal_graph.get_graph_edges():
    node1, node2 = edge.node1, edge.node2
    idx1, idx2 = node_index_map[node1], node_index_map[node2]
    
    # Check edge endpoints to determine directionality
    if edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.TAIL:
        adj_matrix[idx1, idx2] = 1  # Directed edge node1 -> node2
    elif edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.CIRCLE:
        adj_matrix[idx1, idx2] = 1  # Directed edge node1 -> node2

    elif edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW:
        adj_matrix[idx2, idx1] = 1  # Directed edge node2 -> node1
    elif edge.get_endpoint1() == Endpoint.CIRCLE and edge.get_endpoint2() == Endpoint.ARROW:
        adj_matrix[idx2, idx1] = 1  # Directed edge node2 -> node1


GraphDAG(est_dag=adj_matrix.T)
plt.savefig('matteo/fci_graph_adj.png')

pdy = GraphUtils.to_pydot(causal_graph, labels=measurements)
pdy.write_png('matteo/fci_graph.png')