
from data import load_dataset, load_npy_txt_from_file, adjacency_to_digraph

import numpy as np
import pandas as pd
import networkx as nx

from dowhy import gcm

import json
import numpy as np

import matplotlib.pyplot as plt

with open('dataset/ground_truth.json', 'rb') as f:
    data = json.load(f)
n = len(data['adjacency'])
predicted = np.zeros((n,n))
for src, neigh in enumerate(data['adjacency']):
    for nei in neigh:
        dst = int(nei['id'].split('_')[-1])
        predicted[src, dst] = 1

# 1. Modeling cause-effect relationships as a structural causal model
#    (causal graph + functional causal models):

data = load_dataset()
measurements = data.drop(columns='label').columns
# predicted = load_npy_txt_from_file('matteo/merged_adj.txt')

causal_model = gcm.StructuralCausalModel(adjacency_to_digraph(predicted, measurements))  # X -> Y -> Z

gcm.auto.assign_causal_mechanisms(causal_model, data)

# 2. Fitting the SCM to the data:
gcm.fit(causal_model, data)

# Optional: Evaluate causal model
print(gcm.evaluate_causal_model(causal_model, data))

# Step 3: Perform a causal analysis.
# results = gcm.<causal_query>(causal_model, ...
# For instance, root cause analysis:

anomalous_sample = data.iloc[7, :]

# "Which node is the root cause of the anomaly in Z?":
anomaly_attribution = gcm.attribute_anomalies(causal_model, "Station5_mp_85", anomalous_sample)

print(anomaly_attribution)