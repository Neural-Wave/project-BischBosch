import pandas as pd
import networkx as nx
from dowhy import gcm
import numpy as np


def get_data(with_period=False):
    low = pd.read_csv('dataset/low_scrap.csv')
    high = pd.read_csv('dataset/high_scrap.csv')
    if with_period:
        low['period'] = 0
        high['period'] = 1
    return pd.concat([low, high], ignore_index=True)

dataset = get_data()
# sensors = dataset.columns[:-1]


ADJ = np.loadtxt("DAG_MATRIX.txt")
for src in range(ADJ.shape[0]):
    for dst in range(ADJ.shape[1]):
        if ADJ[src, dst] == 1 and ADJ[dst, src] == 1:
            ADJ[src, dst] = 0

g = nx.from_numpy_array(ADJ, create_using=nx.DiGraph())

nx.relabel_nodes(g, {i: col for i, col in enumerate(dataset.columns)} , copy=False)
# print(list(g.nodes(data=True)))
# g = nx.relabel_nodes(g, {i: name for i, name in enumerate(sensors)})
causal_model = gcm.StructuralCausalModel(g)

gcm.auto.assign_causal_mechanisms(causal_model, dataset)

# Step 2: Train our causal model with the data from above:
gcm.fit(causal_model, dataset)

# Step 3: Perform a causal analysis. For instance, root cause analysis, where we observe
anomalous_sample = dataset.iloc[[np.random.randint(0, 99) for _ in range(10)]]  # Here, Y is the root cause.

for col in dataset.columns:
    # ... and would like to answer the question:
    # "Which node is the root cause of the anomaly in Z?":
    anomaly_attribution = gcm.attribute_anomalies(causal_model, col, anomalous_sample, num_distribution_samples=2500)


    # return: A dictionary that assigns a numpy array to each upstream node including the target_node itself. The
    #         i-th entry of an array indicates the contribution of the corresponding node to the anomaly 
    #         score of the target for the i-th observation in anomaly_samples.
    import pickle
    with open(f'Claudio/Pickles/anomalies_{col}.pickle', 'wb') as f:
        pickle.dump(anomaly_attribution, f)

# for k, v in anomaly_attribution.items():
#     print(f'{k}: {[elem for elem in v]}')
