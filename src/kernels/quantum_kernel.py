import pennylane as qml
from src.encoding.angle import AngleEmbedding
from src.utils.device import get_device

dev = get_device("default.qubit", wires=10)
encoder = AngleEmbedding(num_wires=10)

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    # 1. Embed x1
    encoder.apply(x1)
    
    # 2. Entanglement Layer (the "mixer")
    for i in range(9):
        qml.CNOT(wires=[i, i+1])
    # Reverse to ensure full connectivity
    for i in range(8, -1, -1):
        qml.CNOT(wires=[i, i+1])
        
    # 3. Adjoint of x2
    qml.adjoint(lambda: encoder.apply(x2))()
    
    return qml.probs(wires=range(10))