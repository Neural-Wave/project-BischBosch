import json
import numpy as np

import matplotlib.pyplot as plt

with open('dataset/ground_truth.json', 'rb') as f:
    data = json.load(f)

print(data)
print(data['adjacency'])

n = len(data['adjacency'])

adj = np.zeros((n,n))

for src, neigh in enumerate(data['adjacency']):
    for nei in neigh:
        dst = int(nei['id'].split('_')[-1])
        adj[src, dst] = 1

plt.imshow(np.logical_not(adj), cmap='gray')
plt.savefig('fake_gt.png')

