import pennylane as qml
import jax.numpy as jnp
from src.utils.device import get_device
from src.circuits.ansatz import TwoLocalAnsatz

# 1. Setup: Use your custom device manager
dev = get_device("default.qubit", wires=4)
ansatz = TwoLocalAnsatz(num_layers=2, num_wires=4)

# 2. Dummy parameters: 
# Layers * Wires = 2 * 4 = 8 parameters total
params = jnp.array([0.1] * 8)

# 3. Create a QNode: This links your ansatz to the simulator
@qml.qnode(dev)
def circuit(params):
    ansatz.apply(params)
    return qml.expval(qml.PauliZ(0)) # Measure expectation value of qubit 0

# 4. Execute
result = circuit(params)
print(f"Sanity Check Successful!")
print(f"Result of circuit: {result}")