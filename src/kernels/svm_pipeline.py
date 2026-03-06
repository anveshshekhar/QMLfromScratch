import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from src.kernels.matrix_builder import build_kernel_matrix

df = pd.read_csv("mnist_subset.csv")
X = df.drop('label', axis=1).values
y = df['label'].values

subset_size = 50
X_subset = X[:subset_size]
y_subset = y[:subset_size]

print("Building Quantum Kernel Matrix...")
K = build_kernel_matrix(X_subset)

clf = SVC(kernel='precomputed')
clf.fit(K, y_subset)

predictions = clf.predict(K)
accuracy = accuracy_score(y_subset, predictions)

print(f"Quantum Kernel SVM Accuracy: {accuracy * 100:.2f}%")