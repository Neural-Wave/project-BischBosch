import os
os.environ['CASTLE_BACKEND'] = 'pytorch'
import pandas as pd
import numpy as np

from castle.common.priori_knowledge import PrioriKnowledge

import matplotlib.pyplot as plt


def get_data(with_period=False):
    low = pd.read_csv('dataset/low_scrap.csv')
    high = pd.read_csv('dataset/high_scrap.csv')
    if with_period:
        low['period'] = 0
        high['period'] = 1
    return pd.concat([low, high], ignore_index=True)



def load_gt():
    return np.loadtxt('dataset/gt_15_rows.txt')


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
    for src_station in stations.keys():
        for src_sensor in stations[src_station]:
            for dst_station in stations.keys():
                for dst_sensor in stations[dst_station]:
                    if int(src_station[-1]) > int(dst_station[-1]):
                        forbidden.append(
                            (int(src_sensor.split('_')[-1]), int(dst_sensor.split('_')[-1]))
                        )