import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pennylane as qml
from src.circuits.ansatz import TwoLocalAnsatz
from sklearn.decomposition import PCA
from src.encoding.angle import AngleEmbedding
from src.utils.device import get_device
from src.kernels.matrix_builder import build_kernel_matrix
from src.kernels.quantum_kernel import calculate_overlap
# 1. Load and Split Data
df = pd.read_csv("mnist_subset.csv")
X = df.drop('label', axis=1).values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# IMPORTANT: You must fit PCA on train and transform both
pca = PCA(n_components=10)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test) # Use .transform() only for test!

dev = get_device("lightning.qubit", wires=10)
encoder = AngleEmbedding(num_wires=10)
ansatz = TwoLocalAnsatz(num_layers=6, num_wires=10)

@qml.qnode(dev)
def circuit(params, x):
    encoder.apply(x)
    ansatz.apply(params)
    return qml.expval(qml.PauliZ(0))

print("Evaluating Variational Ansatz...")
params = np.load("trained_params.npy")
ansatz_preds = [1 if circuit(params, x) > 0 else 0 for x in X_test_reduced]
acc_ansatz = accuracy_score(y_test, ansatz_preds)
print(f"Ansatz Accuracy: {acc_ansatz * 100:.2f}%")

print("Evaluating Quantum Kernel SVM...")
# 1. Build the TRAIN kernel matrix (N_train x N_train)
K_train = build_kernel_matrix(X_train_reduced)
# Check your kernel matrix stats
print(f"Mean of K_train: {np.mean(K_train):.4f}")
print(f"Std Dev of K_train: {np.std(K_train):.4f}")
print(f"Min/Max: {np.min(K_train):.4f} / {np.max(K_train):.4f}")
# 2. Build the TEST kernel matrix (N_test x N_train)
# You need a function that compares test points against training points
def build_test_kernel_matrix(X_test, X_train):
    n_test = len(X_test)
    n_train = len(X_train)
    test_kernel_matrix = np.zeros((n_test, n_train))
    
    for i in range(n_test):
        for j in range(n_train):
            test_kernel_matrix[i, j] = calculate_overlap(X_test[i], X_train[j])
    return test_kernel_matrix

K_test_final = build_test_kernel_matrix(X_test_reduced, X_train_reduced)

# 3. Train and Predict
clf = SVC(kernel='precomputed')
clf.fit(K_train, y_train)
kernel_preds = clf.predict(K_test_final)
acc_kernel = accuracy_score(y_test, kernel_preds)
print(f"Kernel SVM Accuracy: {acc_kernel * 100:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, ansatz_preds), annot=True, fmt='d', ax=axes[0], cmap='Blues')
axes[0].set_title(f'Variational Ansatz (Acc: {acc_ansatz:.2f})')
sns.heatmap(confusion_matrix(y_test, kernel_preds), annot=True, fmt='d', ax=axes[1], cmap='Greens')
axes[1].set_title(f'Quantum Kernel SVM (Acc: {acc_kernel:.2f})')
plt.show()