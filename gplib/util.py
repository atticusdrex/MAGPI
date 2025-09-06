import jax 
import jax.numpy as jnp
from jax import vmap, value_and_grad
from jax.scipy.linalg import cho_solve, cholesky

from copy import deepcopy
from tqdm import tqdm 
import numpy as np
import math
from jaxopt import LBFGS as JaxoptLBFGS
from scipy.stats import qmc


# 64-bit 
try:
    jax.config.update("jax_enable_x64", True)
except:
    print("64-bit Jax Computation is not available on your CPU.")

def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))

def inv_sigmoid(y):
    return jnp.log(y/(1-y))

def softplus(x):
    return jnp.log(1.0 + jnp.exp(x))

def inv_softplus(y):
    return jnp.log(jnp.exp(y) - 1.0)


def K(X1, X2, kernel, kernel_params):
    return vmap(lambda x: vmap(lambda y: kernel.eval(x, y, kernel_params))(X2))(X1)

# For batching the training data
def create_batches(X, Y, batch_size, shuffle=True):
    n_samples = X.shape[0]
    
    if shuffle:
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        X = X[indices]
        Y = Y[indices]
    
    # Yield batches
    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i + batch_size, :]
        Y_batch = Y[i:i + batch_size]
        yield X_batch, Y_batch

'''
ADAM Optimization Routine
------------------------------------

I do quite a bit of optimizing in these gosh-darn ml scripts and it would be nice to have an encapsulated script for unconstrained ADAM optimization which I could plug into the whole thing instead of rewriting it each time.
'''
def ADAM(
    loss_func, p,
    keys_to_optimize,
    X=jnp.ones((1,1)), 
    Y=jnp.ones((1,1)),
    constr={},
    batch_size=250,
    epochs=100,
    lr=1e-8,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    shuffle=False,
    max_backoff=50
):
    def contains_nan(val_dict):
        return any(jnp.isnan(x).any() for x in val_dict.values())

    def adam_step(m, v, p, grad, lr, t):
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        update = lr * m_hat / (jnp.sqrt(v_hat) + epsilon)
        p = p - update
        return m, v, p

    def try_adam_step(grad_func, p, m, v, lr, t):
        new_p, new_m, new_v = deepcopy(p), {}, {}
        for key in keys_to_optimize:
            m_k, v_k, p_k = adam_step(m[key], v[key], p[key], grad[key], lr, t)
            if key in constr:
                p_k = constr[key](p_k)
            new_p[key], new_m[key], new_v[key] = p_k, m_k, v_k

        # Keep batch inputs
        new_p['X'], new_p['Y'] = p['X'], p['Y']
        loss, grad_new = grad_func(new_p)
        return loss, grad_new, new_p, new_m, new_v

    # Initialize optimizer states
    m = {key: jnp.zeros_like(p[key]) for key in keys_to_optimize}
    v = {key: jnp.zeros_like(p[key]) for key in keys_to_optimize}

    grad_func = value_and_grad(loss_func)

    best_loss = jnp.inf
    best_p = deepcopy(p)

    # Breaking up the training data into batches and storing it in the parameters
    p['X'], p['Y'] = X[:batch_size, :], Y[:batch_size]
    _, grad = grad_func(p)

    iterator = tqdm(range(epochs))

    for epoch in iterator:
        for Xbatch, Ybatch in create_batches(X, Y, batch_size, shuffle=shuffle):
            # Setting the X batches 
            p['X'], p['Y'] = Xbatch, Ybatch

            # Making a trial learning rate 
            trial_lr = lr

            # Backing off learning rate in the case of NaNs found 
            for _ in range(max_backoff):
                loss, grad, trial_p, trial_m, trial_v = try_adam_step(
                    grad_func, p, m, v, trial_lr, epoch+1
                )

                if not (jnp.isnan(loss) or contains_nan(grad)):
                    break  # successful step
                trial_lr *= 0.5
            else:
                print("Too many NaNs. Stopping optimization.")
                return best_p  # return best found so far

            if loss < best_loss:
                best_loss, best_p = loss, deepcopy(trial_p)

            p, m, v = trial_p, trial_m, trial_v

            iterator.set_postfix_str(f"Loss: {loss:.5f}, LR: {trial_lr:.2e}")

    return best_p