
import os
os.environ['CASTLE_BACKEND'] ='pytorch'

import numpy as np
import matplotlib.pyplot as plt

from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, IIDSimulation
from castle.algorithms import DAG_GNN

import pandas as pd

def load_gt():
    return np.loadtxt('dataset/gt_15_rows.txt')



def get_data(with_period=False):
    low = pd.read_csv('dataset/low_scrap.csv')
    high = pd.read_csv('dataset/high_scrap.csv')
    if with_period:
        low['period'] = 0
        high['period'] = 1
    return pd.concat([low, high], ignore_index=True)


if __name__ == "__main__":
    dataset = get_data(with_period=True)

    variable_names = dataset.columns
    stations = {
        'Station1': [col for col in variable_names if col.startswith('Station1')],
        'Station2': [col for col in variable_names if col.startswith('Station2')],
        'Station3': [col for col in variable_names if col.startswith('Station3')],
        'Station4': [col for col in variable_names if col.startswith('Station4')],
        'Station5': [col for col in variable_names if col.startswith('Station5')]
    }

    forbidden = []
    for src_station in stations.keys():
        for src_sensor in stations[src_station]:
            for dst_station in stations.keys():
                for dst_sensor in stations[dst_station]:
                    if int(src_station[-1]) > int(dst_station[-1]):
                        forbidden.append(
                            (int(src_sensor.split('_')[-1]), int(dst_sensor.split('_')[-1]))
                        )

    gt_matrix = load_gt()

    # rl learn
    try:
        gnn = DAG_GNN(
            batch_size=64,
            seed=np.random.randint(0, 100000),
            encoder_hidden=128,
            decoder_hidden=128,
            k_max_iter=20,
            device_type='gpu',
            device_ids='0',
            lr=3e-3,
            graph_threshold=0.1
        )

        dataset_normalized = (dataset - dataset.min()) / (dataset.max() - dataset.min())
        # print(dataset)
        # print(dataset_normalized)
        gnn.learn(
            np.random.permutation(
                dataset_normalized.values
            )
        )
    except KeyboardInterrupt:
        pass

    pred_dag_expert = gnn.causal_matrix[:-1, :-1]

    # 
    # Compare the predicted graph vs the ground truth 
    GraphDAG(
        est_dag=pred_dag_expert
    )
    plt.axvline(85., alpha=0.1)
    uint8_matrix = np.array(pred_dag_expert, dtype=np.uint8)

    np.savetxt(f'Claudio/DAG_matrix_DAGGNN.txt', uint8_matrix, fmt='%d', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
    plt.savefig(f'Claudio/DAG_matrix_DAGGNN.jpg')

    hamming_score = np.count_nonzero(uint8_matrix[:15, :] != gt_matrix)
    print(f"hamming score:  {hamming_score}")

