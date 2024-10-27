
import pickle

from data import load_npy_txt_from_file, save_npy_txt_to_file
from data import filter_adj_mat, load_dataset

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from castle.common import GraphDAG

from causallearn.graph.Endpoint import Endpoint


# with open('matteo/fci_graph.pkl', 'rb') as f:
#     causal_graph = pickle.load(f)
#     # Get list of nodes and create a mapping to indices
#     nodes = causal_graph.get_nodes()
#     node_index_map = {node: idx for idx, node in enumerate(nodes)}

#     # Initialize adjacency matrix
#     num_nodes = len(nodes)
#     fci = np.zeros((num_nodes, num_nodes), dtype=int)

#     # Fill the adjacency matrix based on edge information
#     for edge in causal_graph.get_graph_edges():
#         node1, node2 = edge.node1, edge.node2
#         idx1, idx2 = node_index_map[node1], node_index_map[node2]
        
#         # Check edge endpoints to determine directionality
#         if edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.TAIL:
#             fci[idx1, idx2] = 1  # Directed edge node1 -> node2
#         elif edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.CIRCLE:
#             fci[idx1, idx2] = 1  # Directed edge node1 -> node2

#         elif edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW:
#             fci[idx2, idx1] = 1  # Directed edge node2 -> node1
#         elif edge.get_endpoint1() == Endpoint.CIRCLE and edge.get_endpoint2() == Endpoint.ARROW:
#             fci[idx2, idx1] = 1  # Directed edge node2 -> node1

with open('matteo/lingam_graph.pkl', 'rb') as f:
    causal_matrix = pickle.load(f)
    lingam = causal_matrix.T

with open('matteo/pc_graph.pkl', 'rb') as f:
    causal_graph = pickle.load(f)
    causal_graph.to_nx_graph()
    pc = nx.adjacency_matrix(causal_graph.nx_graph).toarray()

first15_adjmat = load_npy_txt_from_file('dataset/gt_15_rows.txt')
lowert_adjmat = load_npy_txt_from_file('dataset/lower_triangular_gt.txt')
lowert_adjmat[:15] = first15_adjmat

joined = pc
joined[:15] = first15_adjmat

for i in range(joined.shape[0]):
    for j in range(i+1):
        joined[i,j] = lowert_adjmat[i,j]

joined = np.clip(joined, 0, 1)

measurements = load_dataset(shuffle=False).drop(columns='label').columns
joined = filter_adj_mat(joined, measurements)

save_npy_txt_to_file('matteo/merged_adj.txt', joined)

GraphDAG(est_dag=joined)
plt.savefig('matteo/joined_graph_adj.png')
