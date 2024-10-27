
import pickle

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from data import load_dataset, load_npy_txt_from_file

from castle.common import GraphDAG

from causallearn.search.ConstraintBased.PC import pc

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
        edge = gt[i, j]
        if edge == 0:
            bk = bk.add_forbidden_by_node(nodes[i], nodes[j])
        else:
            bk = bk.add_required_by_node(nodes[i], nodes[j])

causal_graph = pc(npy_data, background_knowledge=bk)

with open('matteo/pc_graph.pkl', 'wb') as f:
    pickle.dump(causal_graph, f)

# with open('matteo/pc_graph.pkl', 'rb') as f:
#     causal_graph = pickle.load(f)

causal_graph.to_nx_graph()
GraphDAG(est_dag=nx.adjacency_matrix(causal_graph.nx_graph).toarray())
plt.savefig('matteo/pc_graph_adj.png')

pdy = GraphUtils.to_pydot(causal_graph.G, labels=measurements)
pdy.write_png('matteo/pc_graph.png')