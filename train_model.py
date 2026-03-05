import jax
import pennylane as qml
import pandas as pd
import jax.numpy as jnp
from sklearn.decomposition import PCA
from src.encoding.angle import AngleEmbedding
from src.circuits.ansatz import TwoLocalAnsatz
from src.optimizers.loop import train_step
from src.utils.device import get_device

# 1. Load Data
df = pd.read_csv("mnist_subset.csv")
X = df.drop('label', axis=1).values
y = df['label'].values

# 2. Preprocess: Dimensionality reduction is mandatory for 4-qubit circuits
pca = PCA(n_components=4)
X_reduced = pca.fit_transform(X)

# 3. Setup Quantum Device (Use lightning.qubit for speed)
dev = get_device("lightning.qubit", wires=4)
encoder = AngleEmbedding(num_wires=4)
ansatz = TwoLocalAnsatz(num_layers=2, num_wires=4)

@jax.jit
@qml.qnode(dev)
def circuit(params, x):
    encoder.apply(x)
    ansatz.apply(params)
    return qml.expval(qml.PauliZ(0))

# 4. Training Loop
params = jax.random.normal(jax.random.PRNGKey(0), shape=(8,))
learning_rate = 0.05 

for epoch in range(50):
    total_cost = 0
    # Process sample by sample
    for i in range(len(X_reduced)):
        params, cost = train_step(circuit, params, X_reduced[i], y[i], learning_rate)
        total_cost += cost
        
    print(f"Epoch {epoch} | Average Cost: {total_cost / len(X_reduced):.4f}")