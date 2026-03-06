import jax
import jax.numpy as jnp

def train_step(circuit_func, params, features, labels, learning_rate=0.1):
    def cost(p):
        prediction = circuit_func(p, features)
        return jnp.mean((prediction - labels) ** 2)
    
    grad_fn = jax.grad(cost)
    grads = grad_fn(params)
    new_params = params - (learning_rate * grads)
    return new_params, cost(params)