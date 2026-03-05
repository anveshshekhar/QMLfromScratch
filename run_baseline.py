import pandas as pd
from data.generate_data import generate_mnist_csv
from src.utils.baseline import train_classical_svm, evaluate_baseline
from sklearn.model_selection import train_test_split

# 1. Generate or load the dataset
csv_file = "mnist_subset.csv"
generate_mnist_csv(filename=csv_file)
df = pd.read_csv(csv_file)

# 2. Prepare X (pixels) and y (labels)
X = df.drop('label', axis=1).values
y = df['label'].values

# 3. Split into training and test sets (Crucial for honest benchmarking)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train and Evaluate
model = train_classical_svm(X_train, y_train)
accuracy = evaluate_baseline(model, X_test, y_test)

print(f"Classical RBF-SVM Accuracy on MNIST subset: {accuracy:.4f}")