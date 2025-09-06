from .gp import * 

# Creating a parent class for the multi-fidelity regressor objects 
class MFRegressor:
    """
    Initialize a multi-fidelity object. 

    Parameters
    ----------
    data_dict : dict
        The multi-fidelity data dictionary
    kernel : function
        A kernel covariance function specifying the Gaussian Process prior functional distribution. 
    kernel_dim : int 
        An integer specifying the dimension of the kernel parameters needing to be fed into the kernel function. 
    jitter : float
        A floating point number defining how much to regularize the solution with to avoid numerical instability (default = 1e-6).
    """
    def __init__(self, data_dict, kernel, mean_func, epsilon = 1e-8):
        # Storing data dictionary, kernel and kernel base dimension
        self.d, self.kernel, self.mean = deepcopy(data_dict), kernel, mean_func
        # Storing keyword arguments 
        self.eps = epsilon
        # Number of levels of fidelity
        self.K = len(self.d) 


class Hyperkriging(MFRegressor):
    def __init__(self, *args, tol = 1e-6, max_cond = 1e3, **kwargs):
        # Initializing the parent class 
        super().__init__(*args, **kwargs)

        # Initializing level zero model
        self.d[0]['model'] = GP(self.d[0]['X'], self.d[0]['Y'], self.kernel, self.mean, noise_var = self.d[0]['noise_var'], epsilon = self.eps, max_cond = max_cond, calibrate=True)

        # Initializing models 
        for level in range(1, self.K):
            # Initializing the features 
            features = self.d[level]['X']
            for sublevel in range(level):
                # Getting the mean function from the sublevel immediately under
                mean, _ = self.d[sublevel]['model'].predict(features)

                # Horizontally concatenating the mean function to the existing features 
                features = jnp.hstack((features, mean.reshape(-1,1)))
            
            # Creating a model trained on this set of features 
            self.d[level]['model'] = GP(
                features,
                self.d[level]['Y'],
                self.kernel,
                self.mean,
                noise_var = self.d[level]['noise_var'], epsilon = self.eps, max_cond = max_cond, calibrate=True
            )

    def predict(self, Xtest, level, full_cov = True):
        test_features = Xtest
        # Initializing the features 
        for sublevel in range(level):
            # Getting the mean function from the sublevel immediately under
            mean, _ = self.d[sublevel]['model'].predict(test_features, full_cov = full_cov)

            # Horizontally concatenating the mean function to the existing features 
            test_features = jnp.hstack((test_features, mean.reshape(-1,1)))
        
        return self.d[level]['model'].predict(test_features, full_cov = full_cov)
    
    def optimize(self, level, params = ['k_param', 'm_param', 'noise_var'], lr=1e-3, epochs = 1000, beta = 0.9, k = 15):
        
        # Optimizing lowest-fidelity model 
        if level == 0:
            optimizer = Momentum(
                self.d[0]['model'], 
                neg_mll, beta = beta
            )
            optimizer.kernel_latin_hypercube(k, min=-50, max = 50)
            optimizer.run(lr, epochs, params)
        else:
            features = self.d[level]['X']
            for sublevel in range(level):
                # Getting the mean function from the sublevel immediately under
                mean, _ = self.d[sublevel]['model'].predict(features)

                # Horizontally concatenating the mean function to the existing features 
                features = jnp.hstack((features, mean.reshape(-1,1)))

            # Updating the features at this fidelity-level 
            self.d[level]['model'].X = deepcopy(features)
            
            # Creating a model trained on this set of features 
            optimizer = Momentum(
                self.d[level]['model'], 
                neg_mll, beta = beta
            )
            optimizer.kernel_latin_hypercube(k, min=-50, max = 50)
            optimizer.run(lr, epochs, params)


'''
Kennedy O'Hagan Co-Kriging 
---------------------------------
NOTE: Requires training data to be nested! 
'''
class KennedyOHagan(MFRegressor):
    def __init__(self, *args, tol = 1e-6, max_cond = 1e3, **kwargs):
        # Initializing the parent class 
        super().__init__(*args, **kwargs)

        # Initializing level zero as a simple GP 
        # Initializing level zero model
        self.d[0]['model'] = GP(self.d[0]['X'], self.d[0]['Y'], self.kernel, self.mean, noise_var = self.d[0]['noise_var'], epsilon = self.eps, max_cond = max_cond, calibrate=True)

        # Iterating through the levels of fidelity
        for level in range(1, self.K):
            mean, _ = self.predict(self.d[level]['X'], level-1)
            self.d[level]['model'] = DeltaGP(
                self.d[level]['X'], self.d[level]['Y'], mean.ravel(), self.kernel, self.mean, noise_var = self.d[level]['noise_var'], epsilon = self.eps, max_cond = max_cond, calibrate=True)

    # Update the y2 predictions for each model
    def update(self):
        for level in range(1, self.K):
            mean, _ = self.predict(self.d[level]['X'], level-1)
            self.d[level]['model'].Y2 = mean.ravel()
    
    def predict(self, Xtest, level, full_cov = True):
        # Predicting lowest level of fidelity 
        Ymean, Ycov = self.d[0]['model'].predict(Xtest, full_cov = full_cov)

        # Predicting up to the specified level of fidelity
        for sublevel in range(1, level+1):
            # Getting rho 
            rho = self.d[sublevel]['model'].p['rho']
            # Getting the delta predictions
            delta_mean, delta_cov = self.d[sublevel]['model'].predict(Xtest, full_cov = full_cov)

            # Getting this level's mean and variance 
            Ymean = rho * Ymean + delta_mean
            Ycov = rho**2 * Ycov + delta_cov 

        return Ymean, Ycov 

    def optimize(self, level, params = ['k_param', 'm_param', 'rho', 'noise_var'], lr = 1e-3, epochs = 1000, beta = 0.9, k=15):
        
        # Optimizing lowest-fidelity model 
        if level == 0:
            # Creating a model trained on this set of features 
            optimizer = Momentum(
                self.d[0]['model'], 
                neg_mll, beta = beta
            )
            # Running the optimizer 
            optimizer.kernel_latin_hypercube(k, min=-50, max = 50)
            params.remove("rho")
            optimizer.run(lr, epochs, params)
        else:
            # Updating the levels to approximate 
            for sublevel in range(1, level+1):
                mean, _ = self.predict(self.d[sublevel]['X'], sublevel-1)
                self.d[sublevel]['model'].Y2 = mean.ravel()

            # Creating a model trained on this set of features 
            optimizer = Momentum(
                self.d[level]['model'], 
                delta_neg_mll, beta = beta
            )

            # Running the optimizer 
            optimizer.kernel_latin_hypercube(k, min=-50, max = 50)
            optimizer.run(lr, epochs, params)

class NARGP(MFRegressor):
    def __init__(self, *args, tol = 1e-6, max_cond = 1e3, **kwargs):
        # Initializing the parent class 
        super().__init__(*args, **kwargs)

        # Initializing level zero model
        self.d[0]['model'] = GP(self.d[0]['X'], self.d[0]['Y'], self.kernel, self.mean, noise_var = self.d[0]['noise_var'], epsilon = self.eps, max_cond = max_cond, calibrate=True)

        # Initializing models 
        for level in range(1, self.K):
            # Initializing the features 
            features = self.d[level]['X']
            for sublevel in range(level):
                # Getting the mean function from the sublevel immediately under
                mean, _ = self.d[sublevel]['model'].predict(features)

                # Horizontally concatenating the mean function to the existing features 
                features = jnp.hstack((self.d[level]['X'], mean.reshape(-1,1)))
            
            # Creating a model trained on this set of features 
            self.d[level]['model'] = GP(
                features,
                self.d[level]['Y'],
                self.kernel,
                self.mean,
                noise_var = self.d[level]['noise_var'], epsilon = self.eps, max_cond = max_cond, calibrate=True
            )

    def predict(self, Xtest, level, full_cov = True):
        test_features = Xtest
        # Initializing the features 
        for sublevel in range(level):
            # Getting the mean function from the sublevel immediately under
            mean, _ = self.d[sublevel]['model'].predict(test_features, full_cov = full_cov)

            # Horizontally concatenating the mean function to the existing features 
            test_features = jnp.hstack((Xtest, mean.reshape(-1,1)))
        
        return self.d[level]['model'].predict(test_features, full_cov = full_cov)
    
    def optimize(self, level, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-3, epochs = 1000, beta = 0.9, k = 15):
        
        # Optimizing lowest-fidelity model 
        if level == 0:
            # Creating a model trained on this set of features 
            optimizer = Momentum(
                self.d[level]['model'], 
                neg_mll, beta = beta
            )
            optimizer.kernel_latin_hypercube(k, min=-50, max = 50)
            optimizer.run(lr, epochs, params)
        else:
            features = self.d[level]['X']
            for sublevel in range(level):
                # Getting the mean function from the sublevel immediately under
                mean, _ = self.d[sublevel]['model'].predict(features)

                # Horizontally concatenating the mean function to the existing features 
                features = jnp.hstack((self.d[level]['X'], mean.reshape(-1,1)))

            # Updating the features at this fidelity-level 
            self.d[level]['model'].X = deepcopy(features)
            
            # Creating a model trained on this set of features 
            optimizer = Momentum(
                self.d[level]['model'], 
                neg_mll, beta = beta
            )
            optimizer.kernel_latin_hypercube(k, min=-50, max = 50)
            optimizer.run(lr, epochs, params)