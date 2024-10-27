import os
os.environ['CASTLE_BACKEND'] = 'pytorch'
import pandas as pd

import numpy as np
import networkx as nx

import castle

from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
from castle.algorithms import PC

from castle.common.priori_knowledge import PrioriKnowledge

import matplotlib.pyplot as plt
import json
import numpy as np

import matplotlib.pyplot as plt

with open('dataset/ground_truth.json', 'rb') as f:
    data = json.load(f)

n = len(data['adjacency'])

GROUND_TRUTH = np.zeros((n,n))

for src, neigh in enumerate(data['adjacency']):
    for nei in neigh:
        dst = int(nei['id'].split('_')[-1])
        GROUND_TRUTH[src, dst] = 1

# print(GROUND_TRUTH.sum())
# exit()

def get_data(with_period=False):
    low = pd.read_csv('dataset/low_scrap.csv')
    high = pd.read_csv('dataset/high_scrap.csv')
    if with_period:
        low['period'] = 0
        high['period'] = 1
    return pd.concat([low, high], ignore_index=True)



def load_gt_15_rows():
    return np.loadtxt('dataset/gt_15_rows.txt')

def load_gt_lower():
    return np.loadtxt('dataset/lower_triangular_gt.txt')


if __name__ == "__main__":
    COLORS = [
        '#00B0F0',
        '#FF0000',
        '#B0F000'
    ]

    # Set random seed
    # SEED = 18
    # np.random.seed(SEED)


    dataset = get_data()

    expert_knowledge = PrioriKnowledge(n_nodes=len(dataset.columns))
    variable_names = dataset.columns
    stations = {
        'Station1': [col for col in variable_names if col.startswith('Station1')],
        'Station2': [col for col in variable_names if col.startswith('Station2')],
        'Station3': [col for col in variable_names if col.startswith('Station3')],
        'Station4': [col for col in variable_names if col.startswith('Station4')],
        'Station5': [col for col in variable_names if col.startswith('Station5')]
    }

    forbidden = []
    required = []

    for src_station in stations.keys():
        for src_sensor in stations[src_station]:
            for dst_station in stations.keys():
                for dst_sensor in stations[dst_station]:
                    if int(src_station[-1]) > int(dst_station[-1]):
                        forbidden.append(
                            (int(src_sensor.split('_')[-1]), int(dst_sensor.split('_')[-1]))
                        )

    gt_15_rows_matrix = load_gt_15_rows()

    for src in range(gt_15_rows_matrix.shape[0]):
        for dst in range(gt_15_rows_matrix.shape[1]):
            edge = (src, dst)
            if gt_15_rows_matrix[src][dst] == 1:
                required.append(edge)
            else:
                forbidden.append(edge)

    gt_lower_matrix = load_gt_lower()

    for src in range(gt_lower_matrix.shape[0]):
        for dst in range(src+1):
            edge = (src, dst)
            if gt_lower_matrix[src][dst] == 1:
                required.append(edge)
            else:
                forbidden.append(edge)

    expert_knowledge.add_required_edges(required)
    expert_knowledge.add_forbidden_edges(forbidden)

    # alphas = [0.05, 0.10, 0.15, 0.20]
    # variants = ['original', 'stable', 'parallel']
    # ci_tests = ['fisherz', 'g2', 'chi2']

    # gt = gt_lower_matrix
    # gt[:15] = gt_15_rows_matrix

    alphas = [0.05]
    variants = ['original']
    ci_tests = ['fisherz']

    dataset_normalized = (dataset - dataset.min()) / (dataset.max() - dataset.min())

    for alpha in alphas:
        for variant in variants:
            for ci_test in ci_tests:
                pc = PC(
                    priori_knowledge=expert_knowledge,
                    alpha=alpha,
                    variant=variant,
                    ci_test=ci_test
                )
                
                pc.learn(dataset_normalized.values)
                # Get the predicted graph
                pred_dag_expert = pc.causal_matrix

                # 
                # Compare the predicted graph vs the ground truth 
                GraphDAG(
                    est_dag=pred_dag_expert
                )
                plt.axvline(85., alpha=0.1)
                uint8_matrix = np.array(pred_dag_expert, dtype=np.uint8)

                np.savetxt(f'Claudio/DAG_matrix_{alpha}_{variant}_{ci_test}.txt', uint8_matrix, fmt='%d', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
                plt.savefig(f'Claudio/DAG_matrix_{alpha}_{variant}_{ci_test}.jpg')

                print(f'pred {uint8_matrix.shape}, true {GROUND_TRUTH.shape}')
                hamming_score = np.count_nonzero(uint8_matrix != GROUND_TRUTH)
                print(f"alpha = {alpha}, variant = {variant}, ci_test = {ci_test}:  {hamming_score}")
                # plt.show()


