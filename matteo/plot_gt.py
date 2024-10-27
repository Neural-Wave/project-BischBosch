
from data import load_npy_txt_from_file

import matplotlib.pyplot as plt

from castle.common import GraphDAG

first15_adjmat = load_npy_txt_from_file('dataset/gt_15_rows.txt')
lowert_adjmat = load_npy_txt_from_file('dataset/lower_triangular_gt.txt')
lowert_adjmat[:15] = first15_adjmat

GraphDAG(est_dag=lowert_adjmat)
plt.savefig('matteo/ground_truth.jpg')