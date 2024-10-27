
from data import load_dataset, filter_adj_mat

from data import load_npy_txt_from_file, save_npy_txt_to_file

import matplotlib.pyplot as plt

from castle.common import GraphDAG


dataset = load_dataset(shuffle=True)
measurements = dataset.drop(columns='label').columns

computed_adjmat = load_npy_txt_from_file('.txt')
filtered = filter_adj_mat(computed_adjmat, measurements)

save_npy_txt_to_file('.txt', filtered)

GraphDAG(est_dag=filtered)
plt.savefig('.jpg')
