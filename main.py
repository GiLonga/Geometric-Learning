import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from metrics import dunn_index
from projection import reparametrize_by_arc_length, projection_clock_a_lmbda
from utils import get_max_y_and_roll, curve_unite_length, translate_center_of_mass, rotate_axis

def plot_leaf(training_leaves):
    plt.plot(training_leaves[:,0], training_leaves[:,1])
    plt.axis('equal')
    plt.show()

def load_training_set(workspace_path, i):
    path = os.path.join(workspace_path, f'etape{i}_training.mat')
    A = scipy.io.loadmat(path)
    training_leaves = A[f'etape{i}']
    return training_leaves

path_raw_data =  r"C:\Users\LONGA\Downloads\leaves_parameterized.mat"
workspace_path = os.getcwd()

_dunn = True

#################################################### MAIN #########

n_subdivision = 4
lmbda = 1000
N = 2000
labels_train = np.repeat(np.arange(15), 50)
y_test = np.repeat(np.arange(15), 25)

t_sets = []
dunn_sets = []
for i in range(5):
    set_i = load_training_set(workspace_path, i+1)
    t_sets.append(set_i)
    dunn_sets.append(dunn_index(set_i))
print(dunn_sets)

A_raw = scipy.io.loadmat(path_raw_data)
raw_data = A_raw['leaves_parameterized']

raw_data_reshaped = raw_data.reshape(15, 75, 1000, 2)
training_leaves = raw_data_reshaped[:, :50].reshape(-1, 1000, 2)
testing_leaves  = raw_data_reshaped[:, 50:].reshape(-1, 1000, 2)

starting_etape = training_leaves

new_shape = (starting_etape.shape[0], N, 2)

etape0 = np.zeros(new_shape) # IF WE ARE TRING TO REPARAMETRIZE WITH A DIFFERENT N (NOT 1000) THIS SHOULD BE CHANGED
for index, leaf in enumerate(starting_etape):
    etape0[index] = reparametrize_by_arc_length(leaf, N)[1]
np.save("etape0.npy", etape0)

etape1 = np.zeros(new_shape)
for index, leaf in enumerate(etape0):
    etape1[index] = get_max_y_and_roll(leaf)
    #THIS IS WRONG, YOU ARE JUST ROLLING THEM, WITHOUT SHIFTING.
np.save("etape1.npy", etape1)

etape2 = np.zeros(new_shape)
for index, leaf in enumerate(etape1):
    etape2[index] = curve_unite_length(leaf)
np.save("etape2.npy", etape2)


etape3 = np.zeros(new_shape)
for index, leaf in enumerate(etape2):
    etape3[index] = translate_center_of_mass(leaf)
np.save("etape3.npy", etape3)

etape4 = np.zeros(new_shape)
for index, leaf in enumerate(etape3):
    #step1 = rotate_curve_major_vertical(leaf)
    etape4[index] = rotate_axis(leaf)
np.save("etape4.npy", etape4)

etape5 = np.zeros(new_shape)
etape6 = np.zeros(new_shape)
for index, leaf in enumerate(etape4):
    etape5[index], etape6[index] = projection_clock_a_lmbda(leaf, 4, lmbda) 
    #try:
    #    etape5[index], etape6[index] = projection_clock_a_lmbda(leaf, N, 4)
    #except Exception as e:
    #    print(index)
    #    print(f"There is an error with shape {index} : {e}")
np.save("etape5.npy", etape5)
np.save("etape6.npy", etape6)
test_shape = (375, 2000, 2)

test_0 = np.zeros(test_shape)
for index, leaf in enumerate(testing_leaves):
    test_0[index] = reparametrize_by_arc_length(leaf, N)[1]
np.save("test_0.npy", test_0)

test_1 = np.zeros(test_shape)
for index, leaf in enumerate(test_0):
    test_1[index] = get_max_y_and_roll(leaf)
    #THIS IS WRONG, YOU ARE JUST ROLLING THEM, WITHOUT SHIFT
np.save("test_1.npy", test_1)

test_2 = np.zeros(test_shape)
for index, leaf in enumerate(test_1):
    test_2[index] = curve_unite_length(leaf)
np.save("test_2.npy", test_2)

test_3 = np.zeros(test_shape)
for index, leaf in enumerate(test_2):
    test_3[index] = translate_center_of_mass(leaf)
np.save("test_3.npy", test_3)

test_4 = np.zeros(test_shape)
for index, leaf in enumerate(test_3):
    #step1 = rotate_curve_major_vertical(leaf)
    test_4[index] = rotate_axis(leaf)
np.save("test_4.npy", test_4)

test_5 = np.zeros(test_shape)
test_6 = np.zeros(test_shape)
for index, leaf in enumerate(test_4):
    test_5[index], test_6[index] = projection_clock_a_lmbda(leaf, 4, lmbda) 
np.save("test_5.npy", test_5)
np.save("test_6.npy", test_6)

if _dunn:
    raw_dunn = dunn_index(raw_data)
    print(f'The Dunn index for the raw entire dataset: {raw_dunn}')
    starting_dunn = dunn_index(starting_etape)
    print(f'The Dunn index for the starting dataset: {starting_dunn}')
    etape0_dunn = dunn_index(etape0)
    print(f'The Dunn index for the arc-length parametrization: {etape0_dunn}')
    etape1_dunn = dunn_index(etape1)
    print(f'The Dunn index after centering the curves in the top of their contour: {etape1_dunn}')  
    etape2_dunn = dunn_index(etape2)
    print(f'The Dunn index after normalizing the curves for their length: {etape2_dunn}')
    etape3_dunn = dunn_index(etape3)
    print(f'The Dunn index after centering the curves in their center of mass: {etape3_dunn}')
    etape4_dunn = dunn_index(etape4)
    print(f'The Dunn index after the rotation aligment: {etape4_dunn}')
    etape5_dunn = dunn_index(etape5)
    print(f'The Dunn index for the clock parametrization: {etape5_dunn}')
    etape6_dunn = dunn_index(etape6)
    print(f'The Dunn index for the curvature weighted clock parametrization {etape6_dunn}')


