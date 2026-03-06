import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from src.encoding.angle import AngleEmbedding
from src.circuits.ansatz import TwoLocalAnsatz
from src.optimizers.loop import train_step
from src.utils.device import get_device
import pennylane as qml

df = pd.read_csv("mnist_subset.csv")
X = PCA(n_components=8).fit_transform(df.drop('label', axis=1).values)
X = MinMaxScaler(feature_range=(0, 2 * np.pi)).fit_transform(X)
y = df['label'].values

dev = get_device("lightning.qubit", wires=8)
ansatz = TwoLocalAnsatz(num_layers=4, num_wires=8)

@qml.qnode(dev)
def circuit(params, x):
    AngleEmbedding(num_wires=8).apply(x)
    ansatz.apply(params)
    return qml.expval(qml.PauliZ(0))

num_params = ansatz.num_layers * 8
params = jax.random.normal(jax.random.PRNGKey(0), shape=(num_params,))
learning_rate = 0.02 

print("Starting baseline training-")
for epoch in range(25):
    total_cost = 0
    for i in range(len(X)):
        params, cost = train_step(circuit, params, jnp.array(X[i]), jnp.array(y[i]), learning_rate)
        total_cost += cost
    print(f"Epoch {epoch} -> Average Cost: {total_cost / len(X):.4f}")

np.save("trained_params.npy", np.array(params))