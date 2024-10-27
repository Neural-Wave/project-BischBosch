
import json
import pickle

from data import load_npy_txt_from_file

import numpy as np
import networkx as nx

with open('dataset/ground_truth.json', 'rb') as f:
    data = json.load(f)

n = len(data['adjacency'])
gt_adj = np.zeros((n,n))
for src, neigh in enumerate(data['adjacency']):
    for nei in neigh:
        dst = int(nei['id'].split('_')[-1])
        gt_adj[src, dst] = 1

predicted = load_npy_txt_from_file('matteo/merged_adj.txt')

hamming_score = np.count_nonzero(predicted != gt_adj)
print(f"hamming score:  {hamming_score}")