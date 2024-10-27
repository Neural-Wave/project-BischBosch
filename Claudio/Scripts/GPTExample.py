# Import necessary libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Corrected import statement
from causallearn.search.ScoreBased.notears import notears_linear
from causallearn.utils.GraphUtils import GraphUtils

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Verify 'causal-learn' library version
import causallearn
print("causal-learn version:", causallearn.__version__)

# Step 1: Data Preparation

# Load datasets
low_scrap = pd.read_csv('Low_scrap.csv')
high_scrap = pd.read_csv('High_scrap.csv')

# Add a period indicator
low_scrap['period'] = 0  # Before scrap increase
high_scrap['period'] = 1  # After scrap increase

# Combine datasets
data = pd.concat([low_scrap, high_scrap], ignore_index=True)

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Normalize numerical features
numerical_features = data.select_dtypes(include=['float64', 'int']).columns.tolist()
data[numerical_features] = StandardScaler().fit_transform(data[numerical_features])

# Exclude the 'period' column for initial causal discovery
data_cd = data.drop(columns=['period'])

# Convert DataFrame to NumPy array
data_array = data_cd.values

# Get the list of variable names
variable_names = data_cd.columns.tolist()

# Step 2: Causal Discovery using NOTEARS

# Apply the NOTEARS algorithm
W_est = notears_linear(data_array, lambda1=0.1)

# Enforce temporal constraints

# Define the stations and their variable indices
stations = {
    'Station1': [col for col in variable_names if col.startswith('Station1')],
    'Station2': [col for col in variable_names if col.startswith('Station2')],
    'Station3': [col for col in variable_names if col.startswith('Station3')],
    'Station4': [col for col in variable_names if col.startswith('Station4')],
    'Station5': [col for col in variable_names if col.startswith('Station5')]
}

# Mapping of variable names to indices
var_to_idx = {var: idx for idx, var in enumerate(variable_names)}

# Build temporal mask
num_vars = len(variable_names)
mask = np.zeros((num_vars, num_vars))

stations_list = ['Station1', 'Station2', 'Station3', 'Station4', 'Station5']

for i, src_station in enumerate(stations_list):
    src_indices = [var_to_idx[var] for var in stations[src_station]]
    for dst_station in stations_list[i:]:
        dst_indices = [var_to_idx[var] for var in stations[dst_station]]
        for src_idx in src_indices:
            for dst_idx in dst_indices:
                mask[src_idx, dst_idx] = 1  # Allow edge from src_idx to dst_idx

# Apply mask to enforce temporal constraints
W_est_temporal = W_est * mask

# Threshold the weights to get the adjacency matrix
threshold = 0.3  # Adjust as needed
adjacency_matrix = (np.abs(W_est_temporal) > threshold).astype(int)

# Create a directed graph
G = nx.DiGraph(adjacency_matrix)

# Map indices to variable names
mapping = {i: var for i, var in enumerate(variable_names)}
G = nx.relabel_nodes(G, mapping)

# Remove isolated nodes (if any)
G.remove_nodes_from(list(nx.isolates(G)))

# Step 3: Analyze the Learned Causal Graph

# Visualize the causal graph
plt.figure(figsize=(20, 20))
pos = nx.spring_layout(G, k=0.15, seed=42)
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', arrowsize=20)
plt.title('Causal Graph Learned by NOTEARS')
plt.show()

# Identify variables that have paths to 'Station5_mp_85'
target_variable = 'Station5_mp_85'
predecessors = nx.algorithms.dag.ancestors(G, target_variable)
print("Variables that have paths to", target_variable)
print(predecessors)

# Rank variables based on their shortest path lengths to the target variable
path_lengths = {}
for node in predecessors:
    length = nx.shortest_path_length(G, source=node, target=target_variable)
    path_lengths[node] = length

# Sort variables by path length
sorted_vars = sorted(path_lengths.items(), key=lambda x: x[1])
print("\nVariables ranked by proximity to", target_variable)
for var, length in sorted_vars:
    print(f"{var}: Path length = {length}")

# Step 4: Compare Causal Structures Between Periods

# Prepare data for each period
data_before = data[data['period'] == 0].drop(columns=['period']).values
data_after = data[data['period'] == 1].drop(columns=['period']).values

# Apply NOTEARS for 'before' period
W_est_before = notears_linear(data_before, lambda1=0.1)
W_est_before_temporal = W_est_before * mask
adjacency_matrix_before = (np.abs(W_est_before_temporal) > threshold).astype(int)
G_before = nx.DiGraph(adjacency_matrix_before)
G_before = nx.relabel_nodes(G_before, mapping)
G_before.remove_nodes_from(list(nx.isolates(G_before)))

# Apply NOTEARS for 'after' period
W_est_after = notears_linear(data_after, lambda1=0.1)
W_est_after_temporal = W_est_after * mask
adjacency_matrix_after = (np.abs(W_est_after_temporal) > threshold).astype(int)
G_after = nx.DiGraph(adjacency_matrix_after)
G_after = nx.relabel_nodes(G_after, mapping)
G_after.remove_nodes_from(list(nx.isolates(G_after)))

# Compare the graphs
edges_before = set(G_before.edges())
edges_after = set(G_after.edges())

# Edges that are new in 'after' period
new_edges = edges_after - edges_before
print("\nNew edges in 'after' period:")
print(new_edges)

# Edges that disappeared in 'after' period
removed_edges = edges_before - edges_after
print("\nEdges removed in 'after' period:")
print(removed_edges)

# Analyze changes related to the target variable
new_edges_to_target = [edge for edge in new_edges if edge[1] == target_variable]
print("\nNew edges to", target_variable, "in 'after' period:")
print(new_edges_to_target)

removed_edges_to_target = [edge for edge in removed_edges if edge[1] == target_variable]
print("\nEdges to", target_variable, "removed in 'after' period:")
print(removed_edges_to_target)

# Step 5: Identify and Rank Root Causes

# Mapping of variable names to indices for 'after' period
mapping_inv = {var: idx for idx, var in enumerate(variable_names)}

# Get edge weights for 'after' period
edge_weights_after = {}
for u, v in G_after.edges():
    weight = W_est_after_temporal[mapping_inv[u], mapping_inv[v]]
    edge_weights_after[(u, v)] = weight

# Rank new edges to the target by absolute weight
new_edges_to_target_weights = {edge: edge_weights_after[edge] for edge in new_edges_to_target}
sorted_new_edges = sorted(new_edges_to_target_weights.items(), key=lambda x: -abs(x[1]))
print("\nRanked new edges to", target_variable, "in 'after' period:")
for edge, weight in sorted_new_edges:
    print(f"{edge}: Weight = {weight}")

# Step 6: Visualization

# Create a difference graph
G_diff = nx.DiGraph()
G_diff.add_nodes_from(G_before.nodes())

# Add new edges in 'after' period
G_diff.add_edges_from(new_edges, color='green')

# Add edges removed in 'after' period
G_diff.add_edges_from(removed_edges, color='red')

# Draw the difference graph
edge_colors = [G_diff[u][v]['color'] for u, v in G_diff.edges()]
pos = nx.spring_layout(G_diff, k=0.15, seed=42)
plt.figure(figsize=(20, 20))
nx.draw(G_diff, pos, with_labels=True, node_size=500, node_color='lightblue', arrowsize=20, edge_color=edge_colors)
red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Removed Edges', markerfacecolor='red', markersize=10)
green_patch = plt.Line2D([0], [0], marker='o', color='w', label='New Edges', markerfacecolor='green', markersize=10)
plt.legend(handles=[red_patch, green_patch])
plt.title('Differences in Causal Graphs Between Periods')
plt.show()
