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
    # Return quadratic and log-determinant components
    return 0.5*(quad_term + logdet_term + constant_term)

'''
Negative Marginal Log Likelihood for DeltaGPs 
'''
def delta_neg_mll(model, p, max_cond = 1e2, cond_reg = 1e-2):
    # Getting cholesky factors and solve linear system
    L = model.get_L(p['k_param'], p['noise_var'])
    # Center the Y vector 
    Ytilde = model.Y1 - p['rho'] * model.Y2 - model.mean.eval(model.X, p['m_param']) 
    quad_term = jnp.inner(Ytilde, cho_solve((L, True), Ytilde))
    logdet_term = 2.0*jnp.sum(jnp.log(jnp.diag(L)))
    constant_term = model.N * jnp.log(2*math.pi)
    # Return quadratic and log-determinant components
    return 0.5*(quad_term + logdet_term + constant_term)

def cokriging_neg_mll(model, p):
    L = model.get_L(p)
    # Centering the training data
    Ytilde = model.Yfull - model.mean_train(p)
    # Creating the likelihood terms
    quad_term = jnp.inner(Ytilde, cho_solve((L, True), Ytilde))
    logdet_term = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    constant_term = model.N_total * jnp.log(2*math.pi)
    return 0.5 * (quad_term + logdet_term + constant_term)