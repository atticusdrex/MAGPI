from .likelihood import * 

class Momentum:
    def __init__(self, model, objective_func, beta = 0.9):
        # Storing model and objective function
        self.model, self.obj, self.beta = model, objective_func, beta

        # Storing loss function
        self.grad_fn = value_and_grad(lambda p: self.obj(self.model, p))

        # Compute initial loss
        self.best_loss, _ = self.grad_fn(model.p)
        assert not jnp.isnan(self.best_loss), "Initial Loss is NaN!"

        # Store best params
        self.best_p = deepcopy(model.p)
    
    def kernel_latin_hypercube(self, k, min=-50.0, max=50.0):
        """
        k: number of samples
        min: array of length d with min values
        max: array of length d with max values
        """
        # Getting kernel parameter dimension 
        d = self.model.kernel.p_dim 

        # Getting the dimension of the kernel parameters 
        sampler = qmc.LatinHypercube(d=d)

        # Generate samples in [0,1]^d
        X = sampler.random(n=k*d)

        # Scale to [mins, maxs]
        X_scaled = qmc.scale(X, min*np.ones(d), max*np.ones(d))

        def sample(k_param):
            # Copy model parameters 
            p = deepcopy(self.model.p)

            # Replace kernel hyperparameters
            p['k_param'] = k_param 

            # Evaluate objective function 
            return self.obj(self.model, p)

        # Get the losses at each kernel parameter 
        losses = np.array(vmap(sample)(X_scaled))
        losses[np.isnan(losses)] = 1e99
        best_loss = np.min(losses)

        # Pick the kernel parameters that incurred the lowest loss
        best_p = X_scaled[np.argmin(losses),:]

        # Printing the best loss 
        print("Best Objective Value: %.4e" % (best_loss))

        # Storing the best parameters
        if best_loss < self.best_loss: 
            p = deepcopy(self.model.p)
            p['k_param'] = best_p 
            self.model.set_params(p)

            self.best_loss, self.best_p = best_loss, p



            

    def run(self, lr, epochs, params):
        # Initializing the velocity dictionary 
        v = {} 
        p = deepcopy(self.best_p)
        for param in p.keys():
            v[param] = jnp.zeros_like(p[param])

        # Declaring iterator 
        iterator = tqdm(range(epochs))

        # Storing best loss
        best_loss, best_p = self.best_loss, self.best_p 

        for _ in iterator: 
            # Get value and grad 
            loss, grad = self.grad_fn(p)

            # Gradient-descent with momentum 
            for param in params:
                v[param] = self.beta * v[param] - lr * grad[param]
                p[param] += v[param]
            
            # Display the current loss 
            iterator.set_postfix_str(f'Loss: {loss:.4e}')

            # Store the best loss 
            if loss < best_loss:
                best_loss = best_loss 
                best_p = deepcopy(p) 
        
        # Store the best p as the model aprameters 
        self.model.set_params(best_p) 
        self.best_p = best_p 
            
            
            
        

        

    

class LBFGS:
    def __init__(self, model, objective_func, max_iter=10000, tol=1e-6, max_stepsize=1e-1, verbose = False):
        """
        L-BFGS optimizer wrapper for GP hyperparameter training.
        
        Args:
            model: object with .p (parameters) and .set_params() method
            objective_func: function(model, p) -> scalar loss
            max_iter: maximum number of iterations for solver
            tol: stopping tolerance
        """
        self.model, self.obj = model, objective_func

        # Define scalar loss function in terms of parameters only
        self.loss_fn = lambda p: self.obj(self.model, p)

        # Build JAXOPT solver (handles value_and_grad internally)
        self.solver = JaxoptLBFGS(fun=self.loss_fn, maxiter=max_iter, tol=tol, max_stepsize=max_stepsize, verbose=verbose, history_size = 25, increase_factor = 1.1)

        # Compute initial loss
        self.best_loss = self.loss_fn(model.p)
        assert not jnp.isnan(self.best_loss), "Initial Loss is NaN!"

        # Store best params
        self.best_p = deepcopy(model.p)

    def k_steps(self):
        """
        Run full L-BFGS optimization until convergence.
        Unlike momentum/SGD, we don’t need an epoch loop here:
        jaxopt.LBFGS does the whole solve.
        """
        # Run solver
        res = self.solver.run(self.best_p)

        # Extract results if they're better than what exists
        if res.state.value < self.best_loss:
            best_p, state = res.params, res.state
            best_loss = state.value

            # Save into object
            self.best_p, self.best_loss = best_p, best_loss

        # Print out best log-likelihood
        print("Best objective function eval: %.5f" % (self.best_loss))

        # Update model
        self.model.set_params(self.best_p)


            



        


            
            



            
            
        
                


            

        
