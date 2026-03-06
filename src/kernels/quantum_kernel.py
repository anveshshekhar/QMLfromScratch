import pennylane as qml
from src.encoding.angle import AngleEmbedding
from src.utils.device import get_device

dev = get_device("default.qubit", wires=8)
encoder = AngleEmbedding(num_wires=8)

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    encoder.apply(x1)
    qml.adjoint(encoder.apply)(x2)
    return qml.probs(wires=range(8))

def calculate_overlap(x1, x2):
    return kernel_circuit(x1, x2)[0]