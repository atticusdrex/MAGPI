from .kernel import * 
from .mean import * 
from .optim import * 

class GP:
    '''
    kernel: (class) 
    mean_function: (class)
    '''
    def __init__(self, X, Y, kernel, mean_function, kernel_params = None, mean_params = None, calibrate = True, noise_var = 1e-6, epsilon = 1e-8, max_cond = 1e5):
        # Checking input arguments 
        assert len(X.shape) == 2, "X must be a 2D array (inputs x features)"
        assert len(Y.shape) == 1, "Y must be a 1D array (outputs)"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of data points"
        assert noise_var > 0.0, "White noise variance must be positive real scalar!"
        # Storing training data 
        self.X, self.Y, self.N = X, Y, X.shape[0]
        # Storing input dimension 
        self.input_dim = X.shape[1] 
        # Instantiating and storing kernel covariance function 
        self.kernel = kernel(self.input_dim, epsilon=epsilon)
        # Instantiating and storing the mean function 
        self.mean = mean_function(self.input_dim, epsilon=epsilon)
        # Storing the jitter/epsilon value to avoid singularity and division by zero 
        self.eps = epsilon
        # Initializing parameter dictionary 
        self.p = {
            'noise_var':inv_softplus(noise_var)
        }
        # Storing the kernel parameters of the GP 
        if kernel_params is not None: 
            assert len(kernel_params.shape) == 1, "Kernel parameters must be a 1D array" 
            assert len(kernel_params) == self.kernel.p_dim, "Kernel parameters are wrong dimension (received %d, should be %d)" % (len(kernel_params), self.kernel.p_dim)
            self.p['k_param'] = kernel_params
        else:
            self.p['k_param'] = jnp.ones(self.kernel.p_dim)
        # Storing the mean function parameters of the GP 
        if mean_params is not None: 
            assert len(mean_params.shape) == 1, "Mean function parameters must be a 1D array" 
            assert len(mean_params) == self.mean.p_dim, "Mean function parameters are wrong dimension (received %d, should be %d)" % (len(mean_params), self.mean.p_dim)
            self.p['m_param'] = mean_params 
        else:
            # Setting mean parameters as all zeros 
            self.p['m_param'] = jnp.zeros(self.mean.p_dim)
        # Calibrating parameters if specified
        if calibrate:
            self.p['k_param'] = self.kernel.calibrate(X,Y)
            self.p['m_param'] = self.mean.calibrate(X, Y)
            self.calibrate_noise(max_cond = max_cond)
        # Computing L and alpha values 
        self.L = self.get_L(self.p['k_param'], self.p['noise_var'])
        self.alpha = self.get_alpha(self.L, self.p['m_param'])

    def calibrate_noise(self, max_cond = 1e5):
        # Get condition number 
        L = self.get_L(self.p['k_param'], self.p['noise_var'])
        # Increasing noise to lower condition number 
        while jnp.linalg.cond(L @ L.T) > max_cond:
            self.p['noise_var'] = inv_softplus(1.1*softplus(self.p['noise_var']))
            L = self.get_L(self.p['k_param'], self.p['noise_var'])
        # Printing new noise variance 
        print("Calibrated white noise variance: %.4e" % (softplus(self.p['noise_var'])))

    def set_params(self, p):
        self.p = deepcopy(p)
        self.L = self.get_L(self.p['k_param'], self.p['noise_var'])
        self.alpha = self.get_alpha(self.L, self.p['m_param'])
    
    def get_L(self, k_param, noise_var):
        # Form kernel matrix 
        Ktrain = K(self.X, self.X, self.kernel, k_param) + (self.eps + softplus(noise_var)) * jnp.eye(self.X.shape[0])
        # Take cholesky factorization 
        return cholesky(Ktrain, lower=True)

    def get_alpha(self, L, m_param):
        # Utilize the scipy implementation of cholesky solve
        return cho_solve((L, True), self.Y - self.mean.eval(self.X, m_param))

    def predict(self, Xtest, full_cov = True):
        # Form testing kernel matrix 
        Ktest = K(Xtest, self.X, self.kernel, self.p['k_param'])
        # Compute posterior mean 
        mu = (Ktest @ self.alpha + self.mean.eval(Xtest, self.p['m_param'])).ravel()
        # Returning full covariance or diagonal 
        if full_cov:
            # Computer posterior variance with full testing auxiliary matrix 
            cov = K(Xtest, Xtest, self.kernel, self.p['k_param']) - Ktest @ cho_solve((self.L, True), Ktest.T)
            return mu, cov 
        else:
            # Computer posterior variance without dense test matrix
            Kaux = (jax.vmap(lambda x: self.kernel.eval(x, x, self.p['k_param']))(Xtest)).ravel()
            cov = Kaux - jnp.diag(Ktest @ cho_solve((self.L, True), Ktest.T))
            return mu, cov


class DeltaGP:
    '''
    kernel: (class) 
    mean_function: (class)
    '''
    def __init__(self, X, Y1, Y2, kernel, mean_function, kernel_params = None, mean_params = None, calibrate = True, noise_var = 1e-6, epsilon = 1e-8, max_cond = 1e5):
        # Checking input arguments 
        assert len(X.shape) == 2, "X must be a 2D array (inputs x features)"
        assert len(Y1.shape) == 1, "Y1 must be a 1D array (outputs)"
        assert len(Y2.shape) == 1, "Y2 must be a 1D array (outputs)"
        assert Y1.shape[0] == Y2.shape[0], "Y1 and Y2 must contain the same number of points"
        assert (X.shape[0] == Y1.shape[0]) and (X.shape[0] == Y2.shape[0]), "X and Y must have the same number of data points"
        assert noise_var > 0.0, "White noise variance must be positive real scalar!"
        # Storing training data 
        self.X, self.Y1, self.Y2, self.N = X, Y1, Y2, X.shape[0]
        # Storing input dimension 
        self.input_dim = X.shape[1] 
        # Instantiating and storing kernel covariance function 
        self.kernel = kernel(self.input_dim, epsilon=epsilon)
        # Instantiating and storing the mean function 
        self.mean = mean_function(self.input_dim, epsilon=epsilon)
        # Storing the jitter/epsilon value to avoid singularity and division by zero 
        self.eps = epsilon
        # Initializing parameter dictionary 
        self.p = {
            'rho':jnp.array(1.0),
            'noise_var':inv_softplus(noise_var)
        }
        # Calibrating parameters if specified
        if calibrate:
            self.p['k_param'] = self.kernel.calibrate(X,Y1 - self.p['rho'] * Y2)
            self.p['m_param'] = self.mean.calibrate(X, Y1 - self.p['rho'] * Y2)
            self.calibrate_noise(max_cond = max_cond)
        # Storing the kernel parameters of the GP 
        if kernel_params is not None: 
            assert len(kernel_params.shape) == 1, "Kernel parameters must be a 1D array" 
            assert len(kernel_params) == self.kernel.p_dim, "Kernel parameters are wrong dimension (received %d, should be %d)" % (len(kernel_params), self.kernel.p_dim)
            self.p['k_param'] = kernel_params
        else:
            self.p['k_param'] = jnp.ones(self.kernel.p_dim)
        # Storing the mean function parameters of the GP 
        if mean_params is not None: 
            assert len(mean_params.shape) == 1, "Mean function parameters must be a 1D array" 
            assert len(mean_params) == self.mean.p_dim, "Mean function parameters are wrong dimension (received %d, should be %d)" % (len(mean_params), self.mean.p_dim)
            self.p['m_param'] = mean_params 
        else:
            # Setting mean parameters as all zeros 
            self.p['m_param'] = jnp.zeros(self.mean.p_dim)

        # Computing L and alpha values 
        self.L = self.get_L(self.p['k_param'], self.p['noise_var'])
        self.alpha = self.get_alpha(self.L, self.p['m_param'], self.p['rho'])

    def calibrate_noise(self, max_cond = 1e5):
        # Get condition number 
        L = self.get_L(self.p['k_param'], self.p['noise_var'])
        # Increasing noise to lower condition number 
        while jnp.linalg.cond(L @ L.T) > max_cond:
            self.p['noise_var'] = inv_softplus(1.5*softplus(self.p['noise_var']))
            L = self.get_L(self.p['k_param'], self.p['noise_var'])
        # Printing new noise variance 
        print("Calibrated white noise variance: %.4e" % (softplus(self.p['noise_var'])))

    def set_params(self, p):
        self.p = deepcopy(p)
        self.L = self.get_L(self.p['k_param'], self.p['noise_var'])
        self.alpha = self.get_alpha(self.L, self.p['m_param'], self.p['rho'])
    
    def get_L(self, k_param, noise_var):
        # Form kernel matrix 
        Ktrain = K(self.X, self.X, self.kernel, k_param) + (self.eps + softplus(noise_var)) * jnp.eye(self.X.shape[0])
        # Take cholesky factorization 
        return cholesky(Ktrain, lower=True)
    
    def get_alpha(self, L, m_param, rho):
        # Utilize the scipy implementation of cholesky solve
        return cho_solve((L, True), (self.Y1 - rho * self.Y2) - self.mean.eval(self.X, m_param))

    def predict(self, Xtest, full_cov = False):
        # Form testing kernel matrix 
        Ktest = K(Xtest, self.X, self.kernel, self.p['k_param'])
        # Compute posterior mean 
        mu = (Ktest @ self.alpha + self.mean.eval(Xtest, self.p['m_param'])).ravel()
        
        # Returning full covariance or diagonal 
        if full_cov:
            # Computer posterior variance with full testing auxiliary matrix 
            cov = K(Xtest, Xtest, self.kernel, self.p['k_param']) - Ktest @ cho_solve((self.L, True), Ktest.T)
            return mu, cov 
        else:
            # Computer posterior variance without dense test matrix
            Kaux = (jax.vmap(jit(lambda x: self.kernel.eval(x, x, self.p['k_param'])))(Xtest)).ravel()
            cov = Kaux - jnp.diag(Ktest @ cho_solve((self.L, True), Ktest.T))
            return mu, cov




class SVGP:
    '''
    kernel: (class) 
    mean_function: (class)
    '''
    def __init__(self, X, Y, kernel, mean_function, M = 15, variational_params = None, kernel_params = None, mean_params = None, calibrate = True, noise_var = 1e-6, epsilon = 1e-8, max_cond = 1e5):
        # Checking input arguments 
        assert len(X.shape) == 2, "X must be a 2D array (inputs x features)"
        assert len(Y.shape) == 1, "Y must be a 1D array (outputs)"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of data points"
        assert noise_var > 0.0, "White noise variance must be positive real scalar!"
        # Storing training data 
        self.X, self.Y, self.N, self.M = X, Y, X.shape[0], M
        # Storing input dimension 
        self.input_dim = X.shape[1] 
        # Instantiating and storing kernel covariance function 
        self.kernel = kernel(self.input_dim, epsilon=epsilon)
        # Instantiating and storing the mean function 
        self.mean = mean_function(self.input_dim, epsilon=epsilon)
        # Storing the jitter/epsilon value to avoid singularity and division by zero 
        self.eps = epsilon
        # Greedily choosing the farthest points 
        Z, inds = greedy_k_center(self.X, self.M, seed = 42)
        # Initializing parameter dictionary 
        self.p = {
            'Z':jnp.copy(Z),
            'noise_var':inv_softplus(noise_var)
        }
        # Storing the kernel parameters of the GP 
        if kernel_params is not None: 
            assert len(kernel_params.shape) == 1, "Kernel parameters must be a 1D array" 
            assert len(kernel_params) == self.kernel.p_dim, "Kernel parameters are wrong dimension (received %d, should be %d)" % (len(kernel_params), self.kernel.p_dim)
            self.p['k_param'] = kernel_params
        else:
            self.p['k_param'] = jnp.ones(self.kernel.p_dim)
        # Storing the mean function parameters of the GP 
        if mean_params is not None: 
            assert len(mean_params.shape) == 1, "Mean function parameters must be a 1D array" 
            assert len(mean_params) == self.mean.p_dim, "Mean function parameters are wrong dimension (received %d, should be %d)" % (len(mean_params), self.mean.p_dim)
            self.p['m_param'] = mean_params 
        else:
            # Setting mean parameters as all zeros 
            self.p['m_param'] = jnp.zeros(self.mean.p_dim)
        # Calibrating parameters if specified
        if calibrate:
            self.p['k_param'] = self.kernel.calibrate(Z,Y[inds])
            self.p['m_param'] = self.mean.calibrate(Z, Y[inds])
            self.calibrate_noise(max_cond = max_cond)
        # Storing the variational parameters 
        assert M <= self.N, "Number of inducing points must be less than the number of training data points!"
        # Checking if the user passes a mean and covariance 
        if variational_params is not None: 
            mu, cov = variational_params
            L = jnp.cholesky(cov + self.eps * jnp.eye(M))
        else:
            # Setting the mean to be the Y values at those points 
            mu, L = jnp.copy(self.Y[inds]) - self.mean.eval(Z, self.p['m_param']), jnp.eye(M)*softplus(self.p['noise_var'])

        # Setting the inducing points as the greedily chosen points 
        self.p['q_mu'], self.p['q_L'] = mu, L
        # Computing L and alpha values 
        self.L = self.get_L(self.p['Z'], self.p['k_param'], self.p['noise_var'])
        # Creating constraints for the model 
        xmin, xmax = jnp.min(X[:]), jnp.max(X[:])
        self.constraints = {
            'q_L':lambda L: jnp.tril(L), # keeping L lower-triangular 
            'Z':lambda Z: jnp.clip(Z, xmin, xmax) # Clip the inducing points to the min and max input vals
        }

    def calibrate_noise(self, max_cond = 1e5):
        # Get condition number 
        L = self.get_L(self.p['Z'], self.p['k_param'], self.p['noise_var'])
        # Increasing noise to lower condition number 
        while jnp.linalg.cond(L @ L.T) > max_cond:
            self.p['noise_var'] = inv_softplus(1.1*softplus(self.p['noise_var']))
            L = self.get_L(self.p['Z'], self.p['k_param'], self.p['noise_var'])
        # Printing new noise variance 
        print("Calibrated white noise variance: %.4e" % (softplus(self.p['noise_var'])))

    def set_params(self, p):
        self.p = deepcopy(p)
        self.L = self.get_L(self.p['Z'], self.p['k_param'], self.p['noise_var'])
    
    def get_L(self, Z, k_param, noise_var):
        # Form kernel matrix 
        Ktrain = K(Z, Z, self.kernel, k_param) + (self.eps + softplus(noise_var)) * jnp.eye(self.M)
        # Take cholesky factorization 
        return cholesky(Ktrain, lower=True)

    def predict(self, Xtest, full_samp = True, N_mc = 25, seed = 42):
        # Generating RNG keys 
        keys = jax.random.split(jax.random.key(seed), num = N_mc)

        # Form testing kernel matrix 
        Ktest = K(Xtest, self.p['Z'], self.kernel, self.p['k_param'])

        def single_prediction(key):
            # sampling from variational distribution 
            u = self.p['q_mu'] + self.p['q_L'] @ jax.random.normal(key, shape = (self.M)) 
            # solving the linear system 
            alpha = cho_solve((self.L, True), u) 
            # returning the online prediction
            return (Ktest @ alpha).ravel() + self.mean.eval(Xtest, self.p['m_param'])
        
        # Vector map the single prediction to the keys variable 
        posterior_sample = vmap(single_prediction)(keys)

        if full_samp:
            return posterior_sample 
        else:
            return posterior_sample.mean(axis=1)
