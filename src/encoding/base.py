from abc import ABC, abstractmethod
import pennylane as qml

class QuantumEmbedding(ABC):
    # Abstract Base Class (ABC) for all embedding schemes.
    
    def __init__(self, num_wires):
        self.num_wires = num_wires

    @abstractmethod
    def apply(self, features):
        # Mapping QGates for Classical Features
        
        pass