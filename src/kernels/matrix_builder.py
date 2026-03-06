import numpy as np
from src.kernels.quantum_kernel import calculate_overlap

def build_kernel_matrix(data):
    n_samples = len(data)
    kernel_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i, n_samples):  
            overlap = calculate_overlap(data[i], data[j])
            kernel_matrix[i, j] = overlap
            kernel_matrix[j, i] = overlap  
            
    return kernel_matrix