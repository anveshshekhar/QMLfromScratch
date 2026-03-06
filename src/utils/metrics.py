import jax.numpy as jnp
from sklearn.metrics import accuracy_score

def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, jnp.round(y_pred))

def mean_squared_error(y_true, y_pred):
    return jnp.mean((y_true - y_pred) ** 2)

def compute_loss(predictions, targets, loss_fn='mse'):
    if loss_fn == 'mse':
        return mean_squared_error(targets, predictions)
    else:
        raise ValueError(f"Loss function {loss_fn} not implemented.")