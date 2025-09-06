from .util import * 

'''
Kernel Covariance Functions 
-----------------------------------------
'''
# The parent kernel class which the other kernels inherit 
class Kernel:
    def __init__(self, input_dim, epsilon = 1e-8):
        self.input_dim = input_dim # Storing kernel input dimension 
        self.eps = epsilon # Storing kernel epsilon 

# Automatic Relevancy Determination kernel 
class RBF(Kernel):
    def __init__(self, *args, **kwargs):
        # Calling super class 
        super().__init__(*args, **kwargs)
        # Storing parameter dimension 
        self.p_dim = 1 + self.input_dim
        # Kernel parameter constraints 
        self.constr = lambda params: jnp.maximum(self.eps, params)
    
    def calibrate(self, X, Y):        
        return inv_softplus(self.eps + jnp.concat((jnp.var(Y.ravel()).reshape(1), jnp.var(jnp.diff(Y)) * jnp.ones(self.p_dim-1)), axis=0))
        
    # Evaluation function 
    def eval(self, x, y, params):
        h = (x-y).ravel()
        # Enforcing positivity and boundedness
        params = softplus(params)
        # Computing without constraints 
        return params[0]*jnp.exp(-jnp.sum(h**2 / params[1:]))
    

# Automatic Relevancy Determination kernel 
class NARGP_RBF(Kernel):
    def __init__(self, *args, **kwargs):
        # Calling super class 
        super().__init__(*args, **kwargs)
        # Storing parameter dimension 
        self.p_dim = 2*(1 + self.input_dim)
        # Kernel parameter constraints 
        self.constr = lambda params: jnp.maximum(self.eps, params)
        # Making a list of RBF kernels 
        self.kernels = [RBF(*args, **kwargs), RBF(*args, **kwargs), RBF(1, **kwargs)]
    
    def calibrate(self, X, Y):        
        return inv_softplus(self.eps + jnp.concat((jnp.var(Y.ravel()).reshape(1), jnp.var(jnp.diff(Y)) * jnp.ones(self.p_dim-1)), axis=0))
        
    # Evaluation function 
    def eval(self, x1, x2, params):
        # Extracting the coordinates 
        y, x = x1[-1], x1[:-1]
        yp, xp = x2[-1], x2[:-1]
        # Extracting parameters 
        d = len(x)
        kx, ky, kd = params[:d+1], params[d+1:d+3], params[d+3:]
        # Computing without constraints 
        return self.kernels[0].eval(x, xp, kx) * self.kernels[1].eval(y, yp, ky) + self.kernels[2].eval(x, xp, kd)
    

