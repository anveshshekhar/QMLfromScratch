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
from sklearn.preprocessing import MinMaxScaler
from src.encoding.angle import AngleEmbedding
from src.utils.device import get_device
from src.kernels.matrix_builder import build_kernel_matrix
from src.kernels.quantum_kernel import calculate_overlap

df = pd.read_csv("mnist_subset.csv")
X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1).values, df['label'].values, test_size=0.2, random_state=42)

pca = PCA(n_components=8).fit(X_train)
scaler = MinMaxScaler(feature_range=(0, 2 * np.pi)).fit(pca.transform(X_train))
X_train_scaled = scaler.transform(pca.transform(X_train))
X_test_scaled = scaler.transform(pca.transform(X_test))

dev = get_device("lightning.qubit", wires=8)
ansatz = TwoLocalAnsatz(num_layers=4, num_wires=8)
@qml.qnode(dev)
def circuit(params, x):
    AngleEmbedding(num_wires=8).apply(x)
    ansatz.apply(params)
    return qml.expval(qml.PauliZ(0))

params = np.load("trained_params.npy")
ansatz_preds = [1 if circuit(params, x) > 0 else 0 for x in X_test_scaled]
acc_ansatz = accuracy_score(y_test, ansatz_preds)
print(f"Ansatz Accuracy: {acc_ansatz * 100:.2f}%")

K_train = build_kernel_matrix(X_train_scaled)
print(f"Kernel Matrix Stats | Mean: {np.mean(K_train):.4f} | Std: {np.std(K_train):.4f} | Min/Max: {np.min(K_train):.2f}/{np.max(K_train):.2f}")

K_test = np.array([[calculate_overlap(te, tr) for tr in X_train_scaled] for te in X_test_scaled])
clf = SVC(kernel='precomputed').fit(K_train, y_train)
kernel_preds = clf.predict(K_test)
print(f"Kernel SVM Accuracy: {accuracy_score(y_test, kernel_preds) * 100:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(confusion_matrix(y_test, ansatz_preds), annot=True, fmt='d', ax=axes[0], cmap='Blues')
axes[0].set_title(f'Variational Ansatz (Acc: {acc_ansatz:.2f})')
sns.heatmap(confusion_matrix(y_test, kernel_preds), annot=True, fmt='d', ax=axes[1], cmap='Greens')
axes[1].set_title(f'Quantum Kernel SVM (Acc: {accuracy_score(y_test, kernel_preds):.2f})')
plt.show()

