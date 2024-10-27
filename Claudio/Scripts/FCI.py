from causallearn.search.ConstraintBased.FCI import fci
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import inspect

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
    dataset = get_data()

    variable_names = dataset.columns

    nodes = {
        col: GraphNode(f"x{int(col.split('_')[-1])+1}") for col in variable_names
    }

    stations = {
        'Station1': [col for col in variable_names if col.startswith('Station1')],
        'Station2': [col for col in variable_names if col.startswith('Station2')],
        'Station3': [col for col in variable_names if col.startswith('Station3')],
        'Station4': [col for col in variable_names if col.startswith('Station4')],
        'Station5': [col for col in variable_names if col.startswith('Station5')]
    }

    bk = BackgroundKnowledge()
    # forbidden = []
    for src_station in stations.keys():
        for src_sensor in stations[src_station]:
            for dst_station in stations.keys():
                for dst_sensor in stations[dst_station]:
                    if int(src_station[-1]) > int(dst_station[-1]):
                        bk.add_forbidden_by_node(
                            nodes[src_sensor], 
                            nodes[dst_sensor]
                        )
                        # forbidden.append(
                        #     (int(src_sensor.split('_')[-1]), int(dst_sensor.split('_')[-1]))
                        # )

    # gt_matrix = load_gt()

    # default parameters
    g, edges = fci(dataset.values)
    g.graph[g.graph == 2] = 0
    g.graph[g.graph == -1] = 0

    g.graph = np.logical_not(g.graph)
    print(g.graph)
    # for i in inspect.getmembers(g.__class__):
    #     print(i)

    plt.imshow(np.abs(g.graph), cmap='gray')
    plt.savefig('fci_test.png')

    # visualization
    # from causallearn.utils.GraphUtils import GraphUtils

    # pdy = GraphUtils.to_pydot(g)
    # pdy.write_png('simple_test.png')