import pennylane as qml
from .base import QuantumEmbedding

class AngleEmbedding(QuantumEmbedding):
    def apply(self, features):
        if len(features) > self.num_wires:
            raise ValueError("More features than available wires.")
            
        for i, val in enumerate(features):
            qml.RY(val, wires=i)