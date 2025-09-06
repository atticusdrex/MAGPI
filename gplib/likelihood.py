from .util import * 


'''
Negative Marginal Log Likelihood 
'''
def neg_mll(model, p, max_cond = 1e2, cond_reg = 1e-2):
    # Getting cholesky factors and solve linear system
    L = model.get_L(p['k_param'], p['noise_var'])
    Ytilde = model.Y - model.mean.eval(model.X, p['m_param']) 
    quad_term = jnp.inner(Ytilde, cho_solve((L, True), Ytilde))
    logdet_term = 2.0*jnp.sum(jnp.log(jnp.diag(L)))
    constant_term = model.N * jnp.log(2*math.pi)
    # Regularization component 
    #reg = cond_reg * jnp.maximum(1.0, jnp.log(jnp.linalg.cond(L @ L.T)) - jnp.log(max_cond)) 
    #reg = cond_reg * jnp.maximum(1.0, jnp.log(jnp.max(jnp.diag(L))) - jnp.log(jnp.min(jnp.diag(L))))
    # Return quadratic and log-determinant components
    return 0.5*(quad_term + logdet_term + constant_term) #+ reg

'''
Negative Marginal Log Likelihood for DeltaGPs 
'''
def delta_neg_mll(model, p, max_cond = 1e2, cond_reg = 1e-2):
    # Getting cholesky factors and solve linear system
    L = model.get_L(p['k_param'], p['noise_var'])
    # # approximating the largest and smallest eigenvalues 
    # lambda_min, lambda_max = jnp.max(jnp.diag(L)), jnp.min(jnp.diag(L))
    # # choosing noise variance to get a good condition number 
    # approx_cond = lambda_max / lambda_min 
    # # If the approximate condition number is greater than max_cond then we set the regularization amount 
    # # Compute optimal regularization to make condition number equal to the max cond 
    # alpha = (max_cond * lambda_min - lambda_max) / (1 - max_cond)
    # # Recompute the cholesky factor
    # L = K(model.X, model.X, model.kernel, p['k_param']) + (model.eps + alpha) * jnp.eye(model.X.shape[0])
    # print(alpha)
    # print(L)
    # Center the Y vector 
    Ytilde = model.Y1 - p['rho'] * model.Y2 - model.mean.eval(model.X, p['m_param']) 
    quad_term = jnp.inner(Ytilde, cho_solve((L, True), Ytilde))
    logdet_term = 2.0*jnp.sum(jnp.log(jnp.diag(L)))
    constant_term = model.N * jnp.log(2*math.pi)
    # Regularization component 
    #reg = cond_reg * jnp.maximum(1.0, jnp.log(jnp.linalg.cond(L @ L.T)) - jnp.log(max_cond)) 
    reg = cond_reg * jnp.maximum(1.0, jnp.log(jnp.max(jnp.diag(L))) - jnp.log(jnp.min(jnp.diag(L))))
    # Return quadratic and log-determinant components
    return 0.5*(quad_term + logdet_term + constant_term) + reg