import sys
sys.path.append("..")   # add parent folder (project/) to Python path
from magpi.magpi import *   # now absolute import worksimport math import math 
import matplotlib.pyplot as plt 
import pickle

# Plot parameters 
plt.rcParams.update({
    "font.family": "serif",
    'font.serif': ["Times New Roman"],
    'text.latex.preamble': r'\\usepackage{amsmath}',
    'mathtext.fontset': 'cm',
})

# Define number of fidelity-levels 
n_fidelities = 3 

# Define a list of sample sizes
sample_sizes = [10, 100, 250]

# Define the high and low-fidelity functions
funcs = [
    lambda x: np.exp(-x) * np.sin(2*math.pi*x), 
    lambda x: np.sin(2*math.pi*x),
    lambda x: np.exp(-x)
]

# Making some testing data 
Xtest = np.linspace(0.0, 5.0, 250).reshape(-1,1)

# Loading high-fidelity data
true_data = np.hstack((Xtest, funcs[0](Xtest)))

# Selecting optimal high-fidelity data
np.random.seed(43)
cutoff = 0.0
X_hf = np.random.uniform(cutoff, 5.0, size = sample_sizes[0]).reshape(-1,1)
hf_data = np.hstack((X_hf, funcs[0](X_hf)))

# Loading medium-fidelity data 
X_mf = np.random.uniform(0.0, 5.0, size = sample_sizes[1]).reshape(-1,1)
mf_data = np.hstack((X_mf, funcs[1](X_mf)))

# Loading low-fidelity data 
X_lf = np.random.uniform(0.0, 5.0, size = sample_sizes[2]).reshape(-1,1)
lf_data = np.hstack((X_lf, funcs[2](X_lf)))

# Creating a multi-fidelity-friendly data dictionary 
data_dict = {
    2:{
        'X':hf_data[:,0].reshape(-1,1),
        'Y':hf_data[:,1],
        'noise_var':1e-6,
        'var':np.var(hf_data[:,1]),
        'cost':100.0
    },
    1:{
        'X':mf_data[:,0].reshape(-1,1),
        'Y':mf_data[:,1],
        'noise_var':1e-6,
        'var':np.var(mf_data[:,1]),
        'cost':2.0
    },
    0:{
        'X':lf_data[:,0].reshape(-1,1),
        'Y':lf_data[:,1],
        'noise_var':1e-6,
        'var':np.var(lf_data[:,1]),
        'cost':1.0
    }
}


plt.figure(figsize=(10,4),dpi = 400)
plt.plot(Xtest.ravel(), funcs[0](Xtest).ravel(), linestyle = 'dashed', color = 'black', label = 'High-Fidelity Target Function')
plt.scatter(hf_data[:,0], hf_data[:,1], s = 50, marker = '+', color = 'red', label = 'High-Fidelity Training Data')

plt.xlabel("Input, $\mathbf{x}$")
plt.ylabel("High-Fidelity Function Value")
plt.legend()
plt.xlim([0,5.1])
plt.savefig("results/training-data.png")

# Printing out correlation coefficients 
print(np.corrcoef(funcs[0](Xtest).ravel(), funcs[1](Xtest.ravel()))[0,1])
print(np.corrcoef(funcs[0](Xtest).ravel(), funcs[2](Xtest.ravel()))[0,1])

# Setting number of training steps for each fidelity
n1, n2, n3 = 2500, 10000, 20000

# Training GP-surrogate for lowest level of fidelity 
print("\nTraining MAGPI model...")

# Instantiating the Hyperkriging model 
magpi = MAGPI(
    data_dict, RBF, Linear, max_cond = 1e5, epsilon = 1e-12
)

magpi.optimize(0, params = ['k_param', 'm_param', 'noise_var'], lr = 2e-1, epochs = n1, beta1 = 0.9, beta2=0.999)
magpi.optimize(1, params = ['k_param', 'm_param', 'noise_var'], lr = 2e-1, epochs = n2, beta1 = 0.9, beta2=0.999)
magpi.optimize(2, params = ['k_param', 'm_param', 'noise_var'], lr = 2e-1, epochs = n3, beta1 = 0.9, beta2=0.999)

# Declaring and training Kennedy O'Hagan model 
print("\nTraining Kennedy O'Hagan model...")
koh = KennedyOHagan(
    data_dict, RBF, Linear, max_cond = 1e5, epsilon = 1e-12
)

koh.optimize(0, params = ['k_param', 'm_param', 'rho', 'noise_var'], lr = 2e-1, epochs = n1, beta1 = 0.9, beta2=0.999)
koh.optimize(1, params = ['k_param', 'm_param', 'rho', 'noise_var'], lr = 2e-1, epochs = n2, beta1 = 0.9, beta2=0.999)
koh.optimize(2, params = ['k_param', 'm_param', 'rho', 'noise_var'], lr = 2e-1, epochs = n3, beta1 = 0.9, beta2=0.999)

# Declaring and training NARGP model 
print("\nTraining NARGP model...")
nargp = NARGP(
    data_dict, NARGP_RBF, Linear, max_cond = 1e5, epsilon = 1e-12
)

nargp.optimize(0, params = ['k_param', 'm_param', 'noise_var'], lr = 2e-1, epochs = n1, beta1 = 0.9, beta2=0.999)
nargp.optimize(1, params = ['k_param', 'm_param', 'noise_var'], lr = 2e-1, epochs = n2, beta1 = 0.9, beta2=0.999)
nargp.optimize(2, params = ['k_param', 'm_param', 'noise_var'], lr = 2e-1, epochs = n3, beta1 = 0.9, beta2=0.999)



# Training Single-Fidelity Kriging model 
print("\nTraining Kriging model...")
kr_model = GP(data_dict[2]['X'], data_dict[2]['Y'], RBF, Linear, kernel_params = jnp.ones(2), noise_var = 1e-9, epsilon = 1e-8, max_cond = 1e5, calibrate=True)
optimizer = ADAM(kr_model, neg_mll, beta1=0.9, beta2=0.999)
optimizer.run(1e-2, n2, ['k_param', 'm_param', 'noise_var'])

# Saving the models 
if True:
    with open("models/magpi.pkl", "wb") as outfile:
        pickle.dump(magpi, outfile)
    with open("models/koh.pkl", "wb") as outfile:
        pickle.dump(koh, outfile)
    with open("models/nargp.pkl", "wb") as outfile:
        pickle.dump(nargp, outfile)
    with open("models/kr.pkl", "wb") as outfile:
        pickle.dump(kr_model, outfile)

# Making predictions with each model 
magpi_mean, magpi_cov = magpi.predict(Xtest, 2, full_cov = False)
magpi_conf = 2.00 * jnp.sqrt(magpi_cov)

koh_mean, koh_cov = koh.predict(Xtest, 2, full_cov = False)
koh_conf = 2.00 * jnp.sqrt(koh_cov)

nargp_mean, nargp_cov = nargp.predict(Xtest, 2, full_cov = False)
nargp_conf = 2.00 * jnp.sqrt(nargp_cov)

kr_mean, kr_cov = kr_model.predict(Xtest, full_cov = False) 
kr_conf = 2.00 * np.sqrt(kr_cov)

plt.figure(figsize=(10,6.67),dpi = 400)
plt.subplot(2,2,1)
plt.plot(Xtest.ravel(), funcs[0](Xtest).ravel(), linestyle = 'dashed', color = 'black', label = 'High-Fidelity Target Function')
plt.scatter(hf_data[:,0], hf_data[:,1], s = 50, marker = '+', color = 'red', label = 'High-Fidelity Training Data')
plt.plot(Xtest.ravel(), magpi_mean.ravel(), color = 'green', label = "Proposed Method")
plt.fill_between(Xtest.ravel(), magpi_mean.ravel() - magpi_conf, magpi_mean.ravel() + magpi_conf, color = 'green', alpha = 0.3)
plt.ylabel("High-Fidelity Function Value")
plt.legend()
plt.ylim(-0.6, 1.0)
plt.xlim(0,5)

plt.subplot(2,2,2)
# Kennedy O'Hagan Prediction 
plt.plot(Xtest.ravel(), funcs[0](Xtest).ravel(), linestyle = 'dashed', color = 'black')
plt.scatter(hf_data[:,0], hf_data[:,1], s = 50, marker = '+', color = 'red')
plt.plot(Xtest.ravel(), koh_mean.ravel(), color = 'red', label = "Kennedy O'Hagan")
plt.fill_between(Xtest.ravel(), koh_mean.ravel() - koh_conf, koh_mean.ravel() + koh_conf, color = 'red', alpha = 0.2)
plt.legend()
plt.ylim(-0.6, 1.0)
plt.xlim(0,5)

plt.subplot(2,2,3)
# NARGP Prediction 
plt.plot(Xtest.ravel(), funcs[0](Xtest).ravel(), linestyle = 'dashed', color = 'black')
plt.scatter(hf_data[:,0], hf_data[:,1], s = 50, marker = '+', color = 'red')
plt.plot(Xtest.ravel(), nargp_mean.ravel(), color = 'orange', label = "NARGP")
plt.fill_between(Xtest.ravel(), nargp_mean.ravel() - nargp_conf, nargp_mean.ravel() + nargp_conf, color = 'orange', alpha = 0.3)
plt.legend()
plt.xlabel("Input, $\mathbf{x}$")
plt.ylabel("High-Fidelity Function Value")
plt.ylim(-0.6, 1.0)
plt.xlim(0,5)

# Kriging Prediction 
plt.subplot(2,2,4)
plt.plot(Xtest.ravel(), funcs[0](Xtest).ravel(), linestyle = 'dashed', color = 'black')
plt.scatter(hf_data[:,0], hf_data[:,1], s = 50, marker = '+', color = 'red')
plt.plot(Xtest.ravel(), kr_mean.ravel(), color = 'blue', label = "Single-Fidelity Kriging")
plt.fill_between(Xtest.ravel(), kr_mean.ravel() - kr_conf, kr_mean.ravel() + kr_conf, color = 'blue', alpha = 0.1)
plt.legend()
plt.xlabel("Input, $\mathbf{x}$")
plt.ylim(-0.6, 1.0)
plt.xlim(0,5)
plt.savefig("results/comparison.png")

from sklearn.metrics import mean_squared_error as MSE 
Ytest = funcs[0](Xtest)
print("Method           RMSE         R^2      log MLL")
print("--------------------------------------------------")
print("Proposed Method           \\mathbf{%.3e}   & \\mathbf{%.4f}  &  \\mathbf{%.4f} \\ \\" % (np.sqrt(MSE(Ytest, magpi_mean)), np.corrcoef(Ytest.ravel(), magpi_mean.ravel())[0,1]**2, -neg_mll(magpi.d[2]['model'], magpi.d[2]['model'].p)))
print("Kennedy O'Hagan           %.3e   & %.4f  &  %.4f \\ \\" % (np.sqrt(MSE(Ytest, koh_mean)), np.corrcoef(Ytest.ravel(), koh_mean.ravel())[0,1]**2, -delta_neg_mll(koh.d[2]['model'], koh.d[2]['model'].p)))
print("NARGP                     %.3e   & %.4f  &  %.4f \\ \\" % (np.sqrt(MSE(Ytest, nargp_mean)), np.corrcoef(Ytest.ravel(), nargp_mean.ravel())[0,1]**2, -neg_mll(nargp.d[2]['model'], nargp.d[2]['model'].p)))
print("Kriging                   %.3e   & %.4f  &  %.4f \\ \\" % (np.sqrt(MSE(Ytest, kr_mean)), np.corrcoef(Ytest.ravel(), kr_mean.ravel())[0,1]**2, -neg_mll(kr_model, kr_model.p)))