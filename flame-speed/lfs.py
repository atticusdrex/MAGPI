import sys
sys.path.append("..")   # add parent folder (project/) to Python path
from magpi.magpi import * 
from apak import * 
import pickle 
from sklearn.metrics import mean_squared_error as MSE 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import Ridge

# Plot parameters 
rcParams.update({
    "font.family": "serif",
    'text.latex.preamble': r'\\usepackage{amsmath}',
    'mathtext.fontset': 'cm',
})

if __name__ == "__main__":
    # Making some testing data (edit this to test at different temps)
    test_temp = 850
    Xtest = np.linspace(0.6, 1.4, 250).reshape(-1,1)
    Xtest = np.hstack(
        (Xtest, np.ones((Xtest.shape[0], 1))*test_temp)
    )

    # Levels of fidelity
    K = 5 

    # Temperature vector 
    temps = [450, 550, 650, 750, 850]

    # Initializing data dict
    data_dict = {} 
    for level in range(K):
        data_dict[level] = {
            'X':jnp.array([[]]), 
            'Y':jnp.array([[]])
        }

    # Loading the Cantera data at each temperature profile
    for temp in temps:
        # Loading the data
        with open("data/FlameSpeedData%d.pkl" % temp, 'rb') as infile:
            temp_dict = pickle.load(infile)

        # Iterating through the levels of fidelity
        for level in temp_dict.keys():
            # Obtaining the number of samples
            N = temp_dict[level]['X'].shape[0]
            
            # Getting X and Y matrices
            X, Y = temp_dict[level]['X'], temp_dict[level]['Y']

            # I had two data points that were glitched out with super high LFS values, so I corrected for that here 
            Y[Y > 100] *= 1/700

            # Populating the master data dictionary with the training data at each temperature
            data_dict[level]['X'] = jnp.vstack(
                (data_dict[level]['X'].reshape(-1,2), X)
            )
            data_dict[level]['Y'] = jnp.vstack(
                (data_dict[level]['Y'].reshape(-1,1), jnp.log(Y).reshape(-1,1))
            ).ravel()

    # Filtering out the high-fidelity data 
    test_inds = (data_dict[4]['X'][:,1] == test_temp )
    Xtrue, Ytrue = data_dict[4]['X'][test_inds,:], data_dict[4]['Y'][test_inds]
    inds = (data_dict[4]['X'][:,1] <600)
    data_dict[4]['X'], data_dict[4]['Y'] = data_dict[4]['X'][inds,:], data_dict[4]['Y'][inds]

    # Standard Scaling each set of training and testing inputs 
    from sklearn.preprocessing import StandardScaler 
    scaler = StandardScaler() 
    data_dict[0]['X'] = scaler.fit_transform(data_dict[0]['X'])
    for i in range(1,K):
        data_dict[i]['X'] = scaler.transform(data_dict[i]['X'])
    Xtest = scaler.transform(Xtest)

    # Initializing noise vars
    for level in data_dict.keys():
        data_dict[level]['noise_var'] = 1e-6

    # Declaring MAGPI model 
    print("\nTraining MAGPI model...")
    magpi = MAGPI(
        data_dict, RBF, Linear, max_cond = 1e5, epsilon = 1e-8
    )

    # Training MAGPI model
    magpi.optimize(0, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-1, epochs = 250, beta1 = 0.9, beta2=0.999)
    magpi.optimize(1, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-1, epochs = 1000, beta1 = 0.9, beta2=0.999)
    magpi.optimize(2, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-1, epochs = 2500, beta1 = 0.9, beta2=0.999)
    magpi.optimize(3, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-2, epochs = 5000, beta1 = 0.9, beta2=0.999)
    magpi.optimize(4, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-2, epochs = 11000, beta1 = 0.9, beta2=0.999)

    # Declaring and training Kennedy O'Hagan model 
    print("\nTraining Kennedy O'Hagan model...")
    koh = KennedyOHagan(
        data_dict, RBF, Constant, max_cond = 1e5, epsilon = 1e-12
    )

    koh.optimize(0, params = ['k_param', 'm_param', 'rho', 'noise_var'], lr = 1e-1, epochs = 250, beta1 = 0.9, beta2=0.999)
    koh.optimize(1, params = ['k_param', 'm_param', 'rho', 'noise_var'], lr = 1e-1, epochs = 1000, beta1 = 0.9, beta2=0.999)
    koh.optimize(2, params = ['k_param', 'm_param', 'rho', 'noise_var'], lr = 1e-2, epochs = 2500, beta1 = 0.9, beta2=0.999)
    koh.optimize(3, params = ['k_param', 'm_param', 'rho', 'noise_var'], lr = 1e-2, epochs = 5000, beta1 = 0.9, beta2=0.999)
    koh.optimize(4, params = ['k_param', 'm_param', 'rho', 'noise_var'], lr = 1e-2, epochs = 12500, beta1 = 0.9, beta2=0.999)

    # Declaring and training NARGP model 
    print("\nTraining NARGP model...")
    nargp = NARGP(
        data_dict, NARGP_RBF, Constant, max_cond = 1e5, epsilon = 1e-12
    )

    nargp.optimize(0, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-1, epochs = 250, beta1 = 0.9, beta2=0.999)
    nargp.optimize(1, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-1, epochs = 1000, beta1 = 0.9, beta2=0.999)
    nargp.optimize(2, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-1, epochs = 2500, beta1 = 0.9, beta2=0.999)
    nargp.optimize(3, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-2, epochs = 5000, beta1 = 0.9, beta2=0.999)
    nargp.optimize(4, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-2, epochs = 100000, beta1 = 0.9, beta2=0.999)



    # Training Single-Fidelity Kriging model 
    print("\nTraining Kriging model...")
    kr_model = GP(data_dict[4]['X'], data_dict[4]['Y'], RBF, Constant, kernel_params = jnp.ones(3), noise_var = 1e-9, epsilon = 1e-8, max_cond = 1e5, calibrate=True)
    optimizer = ADAM(kr_model, neg_mll, beta1=0.9, beta2=0.999)
    optimizer.run(1e-2, 25000, ['k_param', 'm_param', 'noise_var'])

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
    magpi_mean, magpi_cov = magpi.predict(Xtest, 4, full_cov = False)
    magpi_conf = 1.96 * jnp.sqrt(magpi_cov)

    koh_mean, koh_cov = koh.predict(Xtest, 4, full_cov = False)
    koh_conf = 1.96 * jnp.sqrt(koh_cov)

    nargp_mean, nargp_cov = nargp.predict(Xtest, 4, full_cov = False)
    nargp_conf = 1.96 * jnp.sqrt(nargp_cov)

    kr_mean, kr_cov = kr_model.predict(Xtest, full_cov = False) 
    kr_conf = 1.96 * np.sqrt(kr_cov)

    import matplotlib.pyplot as plt 

    figure(figsize=(6,6),dpi = 300)

    # Plotting the Hyperkriging predictions with uncertainty estimates
    plot(scaler.inverse_transform(Xtest)[:,0], magpi_mean, color = 'green', label = 'Hyperkriging')
    fill_between(scaler.inverse_transform(Xtest)[:,0], magpi_mean-magpi_conf, magpi_mean + magpi_conf, color = 'green', alpha = 0.3)

    # Plotting the AR1 predictions with uncertainty estimates
    plot(scaler.inverse_transform(Xtest)[:,0], koh_mean, color = 'red', label = 'Kennedy O\'Hagan')
    fill_between(scaler.inverse_transform(Xtest)[:,0], koh_mean-koh_conf, koh_mean + koh_conf, color = 'red', alpha = 0.3)

    # Plotting the NARGP predictions with uncertainty estimates
    plot(scaler.inverse_transform(Xtest)[:,0], nargp_mean, color = 'orange', label = 'NARGP')
    fill_between(scaler.inverse_transform(Xtest)[:,0], nargp_mean-nargp_conf, nargp_mean + nargp_conf, color = 'orange', alpha = 0.3)

    # Plotting the NARGP predictions with uncertainty estimates
    plot(scaler.inverse_transform(Xtest)[:,0], kr_mean, color = 'blue', label = 'Kriging')
    fill_between(scaler.inverse_transform(Xtest)[:,0], kr_mean-kr_conf, kr_mean + kr_conf, color = 'blue', alpha = 0.3)

    # Plotting the Fidelity-3 training data for comparison 
    inds = (scaler.inverse_transform(data_dict[3]['X'])[:,1] == test_temp) 
    Xsim, Ysim = data_dict[3]['X'][inds,:], data_dict[3]['Y'][inds]
    scatter(scaler.inverse_transform(Xsim)[:,0], (Ysim), marker = '.', color = 'black', label = 'Lu 206-Step Mechanism')

    # Plotting the unseen high-fidelity testing data 
    scatter(Xtrue[:,0], (Ytrue), marker = '+', color = 'red', label = "High-Fidelity Testing Data")

    # Plot labeling 
    title("Extrapolative Predictions on Laminar Flame Speed Values at %dK" % (test_temp))
    xlabel("Equivalence Ratio, $\phi$")
    ylabel("Log Laminar Flame Speed - log(m/s)")
    legend()
    savefig("results/LFS_%dK.png" % (test_temp))

    # Fitting a degree-three polynomial through the testing points so we have a more dense error metric
    features = PolynomialFeatures(degree=3)
    lin_model = Ridge(alpha=1e-5)
    lin_model.fit(features.fit_transform(Xtrue[:,0].reshape(-1,1)), Ytrue)
    Yhat = lin_model.predict(features.transform(scaler.inverse_transform(Xtest)[:,0].reshape(-1,1))).ravel()

    lin_model.fit(features.fit_transform(scaler.inverse_transform(Xsim)[:,0].reshape(-1,1)), Ysim)
    Yhat_lu = lin_model.predict(features.transform(scaler.inverse_transform(Xtest)[:,0].reshape(-1,1))).ravel()

    print("Method (%sK)    RMSE         R^2      log MLL" % (test_temp))
    print("--------------------------------------------------")
    print("Hyperkriging:    %.3e &  %.4f &  %.4f \\\\" % (np.sqrt(np.mean(magpi_cov + (Yhat - magpi_mean)**2)), np.corrcoef(Yhat.ravel(), magpi_mean.ravel())[0,1], -neg_mll(magpi.d[4]['model'], magpi.d[4]['model'].p)))
    print("Kennedy O'Hagan: %.3e &  %.4f &  %.4f \\\\" % (np.sqrt(np.mean(koh_cov + (Yhat - koh_mean)**2)), np.corrcoef(Yhat.ravel(), koh_mean.ravel())[0,1], -delta_neg_mll(koh.d[4]['model'], koh.d[4]['model'].p)))
    print("NARGP:           %.3e &  %.4f &  %.4f \\\\" % (np.sqrt(np.mean(nargp_cov + (Yhat - nargp_mean)**2)), np.corrcoef(Yhat.ravel(), nargp_mean.ravel())[0,1], -neg_mll(nargp.d[4]['model'], nargp.d[4]['model'].p)))
    print("Kriging:         %.3e &  %.4f &  %.4f \\\\" % (np.sqrt(np.mean(kr_cov + (Yhat - kr_mean)**2)), np.corrcoef(Yhat.ravel(), kr_mean.ravel())[0,1], -neg_mll(kr_model, kr_model.p)))