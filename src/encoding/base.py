from abc import ABC, abstractmethod
import pennylane as qml

class QuantumEmbedding(ABC):
    def __init__(self, num_wires):
        self.num_wires = num_wires

    @abstractmethod
    def apply(self, features):
        pass