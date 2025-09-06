from .util import * 

'''
Mean Functions 
-------------------------------------------
'''

# The base mean function 
class Mean:
    def __init__(self, input_dim, epsilon=1e-8):
        self.input_dim = input_dim
        self.eps = epsilon

class Linear(Mean):
    def __init__(self, *args, **kwargs):
        # Initializing parent class 
        super().__init__(*args, **kwargs)

        # Storing parameter dimension 
        self.p_dim = 1 + self.input_dim
    
    def calibrate(self, X, Y):
        Phi = jnp.hstack((jnp.ones((X.shape[0],1)), X))
        return jnp.linalg.solve(Phi.T @ Phi + self.eps * jnp.eye(Phi.shape[1]), Phi.T @ Y)
        
    def eval(self, X, params):
        return (params[0] * jnp.ones((X.shape[0], 1)) + X @ params[1:].reshape(-1,1)).ravel()

# Constant mean function
class Constant(Mean):
    def __init__(self, *args, **kwargs):
        # Initializing parent class 
        super().__init__(*args, **kwargs)

        # Storing parameter dimension 
        self.p_dim = 1
    
    def calibrate(self, X, Y):
        return jnp.mean(Y[:]) * jnp.ones(Y.shape[0])
    
    def eval(self, X, params):
        return (params[0] * jnp.ones((X.shape[0], 1))).ravel()

# Zero mean function 
class Zero(Mean):
    def __init__(self, *args, **kwargs):
        # Initializing parent class 
        super().__init__(*args, **kwargs)

        # Storing parameter dimension 
        self.p_dim = self.input_dim
    
    def eval(self, X, params):
        return np.zeros(X.shape[0])