import pandas as pd
from data.generate_data import generate_mnist_csv
from src.utils.baseline import train_classical_svm, evaluate_baseline
from sklearn.model_selection import train_test_split

csv_file = "mnist_subset.csv"
generate_mnist_csv(filename=csv_file)
df = pd.read_csv(csv_file)

X = df.drop('label', axis=1).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = train_classical_svm(X_train, y_train)
accuracy = evaluate_baseline(model, X_test, y_test)

print(f"Classical RBF-SVM Accuracy on MNIST subset: {accuracy:.4f}")