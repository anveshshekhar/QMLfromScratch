import pennylane as qml
import jax.numpy as jnp
from .base import QuantumEmbedding

class AmplitudeEmbedding(QuantumEmbedding):
    def apply(self, features):
        norm = jnp.linalg.norm(features)
        if norm == 0:
            normalized_features = features
        else:
            normalized_features = features / norm
            
        qml.AmplitudeEmbedding(
            features=normalized_features, 
            wires=range(self.num_wires),
            pad_with=0,
            normalize=True
        )