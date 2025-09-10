from normalizer import Preprocess
import numpy as np
import geomstats.backend as gs
import scipy.io
import os
import json
import matplotlib.pyplot as plt
from metrics import dunn_index
from projection import extract_angle_sequence, projection_clock

def plot_leaf(training_leaves):
    plt.plot(training_leaves[:,0], training_leaves[:,1])
    plt.axis('equal')
    plt.show()

#path =  r"C:\Users\LONGA\Downloads\leaves_parameterized.mat"
workspace_path = os.getcwd()
path = os.path.join(workspace_path, 'training_set.mat')
number_of_angles = [4,8,16,32,64]
lmbda = [0.2, 0.5, 1, 10, 100]
A = scipy.io.loadmat(path)
training_leaves = A['etape4']
print(dunn_index(training_leaves))
plot_leaf(training_leaves[450])
n_leaves, n_frames, dim = training_leaves.shape

etape5 = gs.zeros((n_leaves, n_frames, 2))
etape6 = gs.zeros((n_leaves, n_frames, 2))
dunn_matrix_5 = gs.zeros((3,1))
dunn_matrix_6 = gs.zeros((3,1))
for k in range(5):
    for i in range(n_leaves):
        print(i)
        etape5[i], etape6[i] = projection_clock(training_leaves[i], number_of_angles[k], lmbda[k])
    dunn_matrix_5[k] = dunn_index(etape5)
    dunn_matrix_6[k] = dunn_index(etape6)

with open("dunn.json", "w") as f:
    json.dump(dunn_matrix_5.tolist(), f, indent=4) 
print(dunn_matrix_5)
print(dunn_matrix_6)