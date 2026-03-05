import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist

def generate_mnist_csv(digit_a=0, digit_b=1, n_samples=200, filename="mnist_subset.csv"):
    (x_train, y_train), _ = mnist.load_data()
    idx = np.where((y_train == digit_a) | (y_train == digit_b))
    x, y = x_train[idx], y_train[idx]
    indices = np.random.choice(len(x), n_samples, replace=False)
    x_sub, y_sub = x[indices].reshape(n_samples, -1), y[indices]
    df = pd.DataFrame(x_sub)
    df['label'] = y_sub
    df.to_csv(filename, index=False)
    print(f"Successfully generated {filename} with {n_samples} samples.")

if __name__ == "__main__":
    generate_mnist_csv()