
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from visualize import clustering_visualization


########################################################## LEARNING CLASS #####################################################à

def classifier(X_train, y_train, X_test, y_test, index):

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test  = X_test.reshape(X_test.shape[0], -1)
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000, multi_class="auto"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM (RBF kernel)": SVC(kernel="rbf", probability=True, random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print(f"\n🔹 {name}")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=4))
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Save the plot with a unique name
        plt.savefig(f'confusion_matrix_{index}_{name.replace(" ", "_")}.png')
        plt.close() # Close the plot to free up memory


def split_train_test(raw_data, n_classes=15, n_per_class=75, n_train=50, seed=None):
    """
    Split raw leaf data into training and testing sets with random per-class sampling.

    Parameters
    ----------
    raw_data : ndarray of shape (n_classes * n_per_class, ..., ...)
        The dataset, assumed to be ordered such that every consecutive 
        `n_per_class` samples belong to the same class.
    n_classes : int, default=15
        Number of classes in the dataset.
    n_per_class : int, default=75
        Number of samples per class.
    n_train : int, default=50
        Number of training samples to draw randomly per class.
        The remainder (n_per_class - n_train) goes into the test set.
    seed : int or None, default=None
        Random seed for reproducibility. If None, randomness is not fixed.

    Returns
    -------
    training_set : ndarray of shape (n_classes * n_train, ..., ...)
        Training samples.
    testing_set : ndarray of shape (n_classes * (n_per_class - n_train), ..., ...)
        Testing samples.
    """
    if seed is not None:
        np.random.seed(seed)

    # Reshape to group by class
    reshaped = raw_data.reshape(n_classes, n_per_class, *raw_data.shape[1:])

    train_list, test_list = [], []
    for j in range(n_classes):
        perm = np.random.permutation(n_per_class)
        train_idx, test_idx = perm[:n_train], perm[n_train:]
        train_list.append(reshaped[j, train_idx])
        test_list.append(reshaped[j, test_idx])

    training_set = np.vstack(train_list)
    testing_set = np.vstack(test_list)
    return training_set, testing_set

def load_training_set(workspace_path, i):
    path = os.path.join(workspace_path, f'etape{i}_training.mat')
    A = scipy.io.loadmat(path)
    training_leaves = A[f'etape{i}']
    return training_leaves

path_raw_data =  r"C:\Users\LONGA\Downloads\leaves_parameterized.mat"
workspace_path = os.getcwd()



#################################################### MAIN #########

n_subdivision = 4
lmbda = 1000
N = 2000
labels_train = np.repeat(np.arange(15), 50)
y_test = np.repeat(np.arange(15), 25)

A_raw = scipy.io.loadmat(path_raw_data)
raw_data = A_raw['leaves_parameterized']

raw_data_reshaped = raw_data.reshape(15, 75, 1000, 2)
training_leaves = raw_data_reshaped[:, :50].reshape(-1, 1000, 2)
testing_leaves  = raw_data_reshaped[:, 50:].reshape(-1, 1000, 2)

t_sets = []
for i in range(5):
    set_i = load_training_set(workspace_path, i+1)
    t_sets.append(set_i)


for i in range(7):
    etape_name = f"etape{i}.npy"
    test_name = f"test_{i}.npy" 
    etape = np.load(etape_name)
    test = np.load(test_name)
    print (f" Etape number {i}")
    print( classifier(etape, labels_train, test, y_test, i))
    tsne_X = clustering_visualization(etape,labels_train, i)


print ("RAW DATA")
print( classifier(training_leaves, labels_train, testing_leaves, y_test, 999))
