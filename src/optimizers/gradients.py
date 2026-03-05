import jax.numpy as jnp

def parameter_shift_gradient(circuit_func, params, gate_index, shift=jnp.pi/2):
    params_plus = params.at[gate_index].add(shift)
    params_minus = params.at[gate_index].add(-shift)
    
    f_plus = circuit_func(params_plus)
    f_minus = circuit_func(params_minus)
    
    gradient = 0.5 * (f_plus - f_minus)
    
    return gradient