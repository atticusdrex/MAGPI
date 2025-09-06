from gplib.mf import * 
import pickle 
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Making some testing data (edit this to test at different temps)
    test_temp = 650
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
        with open("LaminarFlameSpeed/data/FlameSpeedData%d.pkl" % temp, 'rb') as infile:
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
                (data_dict[level]['Y'].reshape(-1,1), Y.reshape(-1,1))
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
    
    # Defining number of low and high-fidelity iterations 
    lf_iter, hf_iter, max_stepsize, max_cond = 5000, 10000, 1e-3, 5e5

    # Training Hyperkriging model 
    print("\nTraining Hyperkriging Model...")
    model = Hyperkriging(
        data_dict, RBF, Linear, max_cond = max_cond, epsilon = 1e-12
    )

    # Training GP-Surrogates at each level of fidelity
    model.optimize(0, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-3, epochs = 250, beta = 0.9, k = 50)
    model.optimize(1, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-3, epochs = 250, beta = 0.9, k = 50)
    model.optimize(2, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-3, epochs = 1000, beta = 0.9, k = 50)
    model.optimize(3, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-3, epochs = 2500, beta = 0.9, k = 50)
    model.optimize(4, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-4, epochs = 5000, beta = 0.9, k = 50)

    # Training KOH model 
    print("\nTraining Kennedy O'Hagan model...")
    koh = KennedyOHagan(
        data_dict, RBF, Linear, max_cond = max_cond, epsilon = 1e-12
    )

    koh.optimize(0, params = ['k_param', 'm_param', 'rho', 'noise_var'], lr = 1e-3, epochs = 250, beta = 0.9, k = 50)
    koh.optimize(0, params = ['k_param', 'm_param', 'rho', 'noise_var'], lr = 1e-3, epochs = 250, beta = 0.9, k = 50)
    koh.optimize(0, params = ['k_param', 'm_param', 'rho', 'noise_var'], lr = 1e-3, epochs = 250, beta = 0.9, k = 50)
    koh.optimize(0, params = ['k_param', 'm_param', 'rho', 'noise_var'], lr = 1e-3, epochs = 1000, beta = 0.9, k = 50)
    koh.optimize(0, params = ['k_param', 'm_param', 'rho', 'noise_var'], lr = 1e-3, epochs = 2500, beta = 0.9, k = 50)

    # Training NARGP model
    print("\nTraining NARGP model...")
    nargp = NARGP(
        data_dict, NARGP_RBF, Linear, max_cond = max_cond, epsilon = 1e-16
    )

    # Training GP-Surrogates at each level of fidelity
    nargp.optimize(0, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-3, epochs = 250, beta = 0.9, k = 50)
    nargp.optimize(1, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-3, epochs = 250, beta = 0.9, k = 50)
    nargp.optimize(2, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-3, epochs = 1000, beta = 0.9, k = 50)
    nargp.optimize(3, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-3, epochs = 1000, beta = 0.9, k = 50)
    nargp.optimize(4, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-3, epochs = 2500, beta = 0.9, k = 50)

    # Training Single-Fidelity Kriging model 
    print("\nTraining Kriging model...")
    kr_model = GP(data_dict[2]['X'], data_dict[2]['Y'], RBF, Linear, kernel_params = jnp.ones(2), noise_var = 1e-9, epsilon = 1e-8, max_cond = 1e5, calibrate=True)
    optimizer = Momentum(kr_model, neg_mll, beta = 0.9)
    optimizer.kernel_latin_hypercube(15, min=-50, max = 50)
    optimizer.run(1e-3, 1000, ['k_param', 'm_param', 'noise_var'])

    # Making predictions with each model 
    hk_mean, hk_cov = model.predict(Xtest, 4, full_cov = False)
    hk_conf = 1.96 * jnp.sqrt(hk_cov)

    koh_mean, koh_cov = koh.predict(Xtest, 4, full_cov = False)
    koh_conf = 1.96 * jnp.sqrt(koh_cov)

    nargp_mean, nargp_cov = nargp.predict(Xtest, 4, full_cov = False)
    nargp_conf = 1.96 * jnp.sqrt(nargp_cov)

    kr_mean, kr_cov = kr_model.predict(Xtest, full_cov = False) 
    kr_conf = 1.96 * np.sqrt(kr_cov)

    # Plotting Results
    plt.figure(figsize=(15,6),dpi = 500)

    # Plotting the Hyperkriging predictions with uncertainty estimates
    plt.plot(scaler.inverse_transform(Xtest)[:,0], hk_mean, color = 'green', label = 'Hyperkriging')
    plt.fill_between(scaler.inverse_transform(Xtest)[:,0], hk_mean-hk_conf, hk_mean + hk_conf, color = 'green', alpha = 0.3)

    # Plotting the AR1 predictions with uncertainty estimates
    plt.plot(scaler.inverse_transform(Xtest)[:,0], koh_mean, color = 'red', label = 'Kennedy O\'Hagan')
    plt.fill_between(scaler.inverse_transform(Xtest)[:,0], koh_mean-koh_conf, koh_mean + koh_conf, color = 'red', alpha = 0.3)

    # Plotting the NARGP predictions with uncertainty estimates
    plt.plot(scaler.inverse_transform(Xtest)[:,0], nargp_mean, color = 'orange', label = 'NARGP')
    plt.fill_between(scaler.inverse_transform(Xtest)[:,0], nargp_mean-nargp_conf, nargp_mean + nargp_conf, color = 'orange', alpha = 0.3)

    # # Plotting the NARGP predictions with uncertainty estimates
    # plt.plot(scaler.inverse_transform(Xtest)[:,0], kr_mean, color = 'blue', label = 'Kriging')
    # plt.fill_between(scaler.inverse_transform(Xtest)[:,0], kr_mean-kr_conf, kr_mean + kr_conf, color = 'blue', alpha = 0.3)

    # Plotting the Fidelity-3 training data for comparison 
    inds = (scaler.inverse_transform(data_dict[3]['X'])[:,1] == test_temp) 
    Xsim, Ysim = data_dict[3]['X'][inds,:], data_dict[3]['Y'][inds]
    plt.scatter(scaler.inverse_transform(Xsim)[:,0], (Ysim), marker = '.', color = 'black', label = 'Lu 206-Step Mechanism')

    # Plotting the unseen high-fidelity testing data 
    plt.scatter(Xtrue[:,0], (Ytrue), marker = '+', color = 'red', label = "High-Fidelity Testing Data")

    # Plot labeling 
    plt.title("Extrapolative Predictions on Laminar Flame Speed Values at %dK" % (test_temp))
    plt.xlabel("Equivalence Ratio, $\phi$")
    plt.ylabel("Log Laminar Flame Speed - log(m/s)")
    plt.legend()
    plt.savefig("LaminarFlameSpeed/figures/Results_%dK.png" % (test_temp))
    


