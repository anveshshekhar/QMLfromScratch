import jax
import jax.numpy as jnp

def train_step(circuit_func, params, features, labels, learning_rate=0.1):
    """
    Perform a single step of gradient descent using JAX's native auto-diff.
    """
    # 1. Define cost function
    def cost(p):
        prediction = circuit_func(p, features)
        return jnp.mean((prediction - labels) ** 2)
    
    # 2. Use JAX native gradient (much faster and JIT compatible)
    grad_fn = jax.grad(cost)
    grads = grad_fn(params)
    
    # 3. Update parameters
    new_params = params - (learning_rate * grads)
    return new_params, cost(params)