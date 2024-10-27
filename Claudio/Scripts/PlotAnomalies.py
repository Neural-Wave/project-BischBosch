import pickle
import pandas as pd
import networkx as nx
from dowhy import gcm
import numpy as np


sensor = "Station3_mp_40"

with open('Claudio/Pickles/anomalies_{sensor}.pickle', 'rb') as f:
    anomaly_attribution = pickle.load(f)

for k, v in anomaly_attribution.items():
    print(f'{k}: {v}')