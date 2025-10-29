import sys
sys.path.append("..")   # add parent folder (project/) to Python path
from gplib.mf import * 
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error as MSE 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import Ridge

plt.rcParams.update({
    "font.family": "serif",
    'text.latex.preamble': r'\\usepackage{amsmath}',
    'mathtext.fontset': 'cm',
})

if __name__ == "__main__":
    # Loading the models 
    with open("models/hk.pkl", "rb") as infile:
        hk = pickle.load(infile)
    with open("models/koh.pkl", "rb") as infile:
        koh = pickle.load(infile)
    with open("models/nargp.pkl", "rb") as infile:
        nargp = pickle.load(infile)
    with open("models/kr.pkl", "rb") as infile:
        kr = pickle.load(infile)   
    # Setting up the plot parameters 
    plt.figure(figsize=(9,9),dpi = 300)
    plt.subplot(2,2,1)

    # Iterating through the test temps 
    test_temps = [550, 650, 750, 850]
    for plot_num, test_temp in enumerate(test_temps):
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
        test_inds = (data_dict[4]['X'][:,1] == test_temp)
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
            data_dict[level]['noise_var'] = 1e-2
        
        hk_mean, hk_cov = hk.predict(Xtest, 4, full_cov = False)
        hk_conf = 1.96 * jnp.sqrt(hk_cov)

        koh_mean, koh_cov = koh.predict(Xtest, 4, full_cov = False)
        koh_conf = 1.96 * jnp.sqrt(koh_cov)

        nargp_mean, nargp_cov = nargp.predict(Xtest, 4, full_cov = False)
        nargp_conf = 1.96 * jnp.sqrt(nargp_cov)

        kr_mean, kr_cov = kr.predict(Xtest, full_cov = False) 
        kr_conf = 1.96 * np.sqrt(kr_cov)

        # Plotting the data 
        plt.subplot(2,2,plot_num+1)
        # Plotting the Hyperkriging predictions with uncertainty estimates
        plt.plot(scaler.inverse_transform(Xtest)[:,0], hk_mean, color = 'green', label = 'Our Method')
        plt.fill_between(scaler.inverse_transform(Xtest)[:,0], hk_mean-hk_conf, hk_mean + hk_conf, color = 'green', alpha = 0.3)

        # Plotting the AR1 predictions with uncertainty estimates
        plt.plot(scaler.inverse_transform(Xtest)[:,0], koh_mean, color = 'red', label = 'Kennedy O\'Hagan')
        plt.fill_between(scaler.inverse_transform(Xtest)[:,0], koh_mean-koh_conf, koh_mean + koh_conf, color = 'red', alpha = 0.3)

        # Plotting the NARGP predictions with uncertainty estimates
        plt.plot(scaler.inverse_transform(Xtest)[:,0], nargp_mean, color = 'orange', label = 'NARGP')
        plt.fill_between(scaler.inverse_transform(Xtest)[:,0], nargp_mean-nargp_conf, nargp_mean + nargp_conf, color = 'orange', alpha = 0.3)

        # Plotting the NARGP predictions with uncertainty estimates
        plt.plot(scaler.inverse_transform(Xtest)[:,0], kr_mean, color = 'blue', label = 'Kriging')
        plt.fill_between(scaler.inverse_transform(Xtest)[:,0], kr_mean-kr_conf, kr_mean + kr_conf, color = 'blue', alpha = 0.3)

        # Plotting the Fidelity-3 training data for comparison 
        inds = (scaler.inverse_transform(data_dict[3]['X'])[:,1] == test_temp) 
        Xsim, Ysim = data_dict[3]['X'][inds,:], data_dict[3]['Y'][inds]
        plt.scatter(scaler.inverse_transform(Xsim)[:,0], (Ysim), marker = '.', color = 'black', label = 'Lu 206-Step Mechanism')

        # Plotting the unseen high-fidelity testing data 
        plt.scatter(Xtrue[:,0], Ytrue, marker = '+', color = 'red', label = "High-Fidelity Testing Data")
        # Plot labeling 
        plt.title("Predictions at %dK" % (test_temp))
        if plot_num > 1:
            plt.xlabel("Equivalence Ratio, $\phi$")
        if plot_num == 0 or plot_num == 2:
            plt.ylabel("Log Laminar Flame Speed - log(m/s)")
        if plot_num == 0:
            plt.legend()

        # Fitting a degree-three polynomial through the testing points so we have a more dense error metric
        features = PolynomialFeatures(degree=3)
        lin_model = Ridge(alpha=1e-5)
        lin_model.fit(features.fit_transform(Xtrue[:,0].reshape(-1,1)), Ytrue)
        Yhat = lin_model.predict(features.transform(scaler.inverse_transform(Xtest)[:,0].reshape(-1,1))).ravel()

        lin_model.fit(features.fit_transform(scaler.inverse_transform(Xsim)[:,0].reshape(-1,1)), Ysim)
        Yhat_lu = lin_model.predict(features.transform(scaler.inverse_transform(Xtest)[:,0].reshape(-1,1))).ravel()

        print("Method (%sK)    RMSE         R^2      log MLL" % (test_temp))
        print("--------------------------------------------------")
        print("Hyperkriging:    %.3e &  %.4f &  %.4f \\\\" % (np.sqrt(np.mean(hk_cov + (Yhat - hk_mean)**2)), np.corrcoef(Yhat.ravel(), hk_mean.ravel())[0,1], -neg_mll(hk.d[4]['model'], hk.d[4]['model'].p)))
        print("Kennedy O'Hagan: %.3e &  %.4f &  %.4f \\\\" % (np.sqrt(np.mean(koh_cov + (Yhat - koh_mean)**2)), np.corrcoef(Yhat.ravel(), koh_mean.ravel())[0,1], -delta_neg_mll(koh.d[4]['model'], koh.d[4]['model'].p)))
        print("NARGP:           %.3e &  %.4f &  %.4f \\\\" % (np.sqrt(np.mean(nargp_cov + (Yhat - nargp_mean)**2)), np.corrcoef(Yhat.ravel(), nargp_mean.ravel())[0,1], -neg_mll(nargp.d[4]['model'], nargp.d[4]['model'].p)))
        print("Kriging:         %.3e &  %.4f &  %.4f \\\\" % (np.sqrt(np.mean(kr_cov + (Yhat - kr_mean)**2)), np.corrcoef(Yhat.ravel(), kr_mean.ravel())[0,1], -neg_mll(kr, kr.p)))
    
    plt.savefig("results/composite.png")

    
