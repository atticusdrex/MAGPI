import sys
sys.path.append("..")   # add parent folder (project/) to Python path
from jaxgp.mf import * 
from apak import * 
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error as MSE 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import Ridge

plt.rcParams.update({
    "font.family": "serif",
    'font.serif': ["Times New Roman"],
    'text.latex.preamble': r'\\usepackage{amsmath}',
    'mathtext.fontset': 'cm',
})

if __name__ == "__main__":
    # Loading the models 
    with open("models/magpi.pkl", "rb") as infile:
        magpi = pickle.load(infile)
    with open("models/koh.pkl", "rb") as infile:
        koh = pickle.load(infile)
    with open("models/nargp.pkl", "rb") as infile:
        nargp = pickle.load(infile)
    with open("models/kr.pkl", "rb") as infile:
        kr = pickle.load(infile)   
    # Setting up the plot parameters 
    fig, axs = plt.subplots(4,4,figsize=(10,10),dpi = 300)
    # plt.subplot(4,4,1)

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
        # Saving data to plot
        if plot_num == 3: 
            vis_inds = ((data_dict[4]['X'][:,1] <= test_temp) & (data_dict[4]['X'][:,1] > 600))
            Xvis, Yvis = data_dict[4]['X'][vis_inds,:], data_dict[4]['Y'][vis_inds]

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
        
        magpi_mean, magpi_cov = magpi.predict(Xtest, 4, full_cov = False)
        magpi_conf = 2.00 * jnp.sqrt(magpi_cov)

        koh_mean, koh_cov = koh.predict(Xtest, 4, full_cov = False)
        koh_conf = 2.00 * jnp.sqrt(koh_cov)

        nargp_mean, nargp_cov = nargp.predict(Xtest, 4, full_cov = False)
        nargp_conf = 2.00 * jnp.sqrt(nargp_cov)

        kr_mean, kr_cov = kr.predict(Xtest, full_cov = False) 
        kr_conf = 2.00 * np.sqrt(kr_cov)

        # Plotting the data 
        ax = plt.subplot(4,4,4*plot_num+1)
        # Plotting the MAGPI predictions with uncertainty estimates
        plt.plot(scaler.inverse_transform(Xtest)[:,0], magpi_mean, color = 'green', label = 'Proposed Method')
        plt.fill_between(scaler.inverse_transform(Xtest)[:,0], magpi_mean-magpi_conf, magpi_mean + magpi_conf, color = 'green', alpha = 0.3)
        # Plotting the Fidelity-3 training data for comparison 
        inds = (scaler.inverse_transform(data_dict[3]['X'])[:,1] == test_temp) 
        Xsim, Ysim = data_dict[3]['X'][inds,:], data_dict[3]['Y'][inds]
        plt.scatter(scaler.inverse_transform(Xsim)[:,0], (Ysim), marker = '.', color = 'black', label = 'Lu 206-Step Mechanism')
        # Plotting the unseen high-fidelity testing data 
        plt.scatter(Xtrue[:,0], Ytrue, marker = '+', color = 'red', label = "High-Fidelity Testing Data")
        if plot_num + 1 == 1:
            plt.title("Proposed Method")
        plt.ylabel("$T_0=%dK$" % (test_temp), fontsize=14)
        if plot_num + 1 < 4:
            ax.set_xticks([])         # Removes all tick marks
            ax.set_xlabel('')         # Removes the axis label

        ax = plt.subplot(4,4,4*plot_num+2)
        # Plotting the AR1 predictions with uncertainty estimates
        plt.plot(scaler.inverse_transform(Xtest)[:,0], koh_mean, color = 'red', label = 'Kennedy O\'Hagan')
        plt.fill_between(scaler.inverse_transform(Xtest)[:,0], koh_mean-koh_conf, koh_mean + koh_conf, color = 'red', alpha = 0.3)
        # Plotting the Fidelity-3 training data for comparison 
        inds = (scaler.inverse_transform(data_dict[3]['X'])[:,1] == test_temp) 
        Xsim, Ysim = data_dict[3]['X'][inds,:], data_dict[3]['Y'][inds]
        plt.scatter(scaler.inverse_transform(Xsim)[:,0], (Ysim), marker = '.', color = 'black', label = 'Lu 206-Step Mechanism')
        # Plotting the unseen high-fidelity testing data 
        plt.scatter(Xtrue[:,0], Ytrue, marker = '+', color = 'red', label = "High-Fidelity Testing Data")
        if plot_num + 1 == 1:
            plt.title("Kennedy O'Hagan")
        if plot_num + 1 < 4:
            ax.set_xticks([])         # Removes all tick marks
            ax.set_xlabel('')         # Removes the axis label
        ax.set_yticks([])         # Removes all tick marks
        ax.set_ylabel('')         # Removes the axis label
        
        ax = plt.subplot(4,4,4*plot_num+3)
        # Plotting the NARGP predictions with uncertainty estimates
        plt.plot(scaler.inverse_transform(Xtest)[:,0], nargp_mean, color = 'orange', label = 'NARGP')
        plt.fill_between(scaler.inverse_transform(Xtest)[:,0], nargp_mean-nargp_conf, nargp_mean + nargp_conf, color = 'orange', alpha = 0.3)
        # Plotting the Fidelity-3 training data for comparison 
        inds = (scaler.inverse_transform(data_dict[3]['X'])[:,1] == test_temp) 
        Xsim, Ysim = data_dict[3]['X'][inds,:], data_dict[3]['Y'][inds]
        plt.scatter(scaler.inverse_transform(Xsim)[:,0], (Ysim), marker = '.', color = 'black', label = 'Lu 206-Step Mechanism')
        # Plotting the unseen high-fidelity testing data 
        plt.scatter(Xtrue[:,0], Ytrue, marker = '+', color = 'red', label = "High-Fidelity Testing Data")
        if plot_num + 1 == 1:
            plt.title("NARGP")
        if plot_num + 1 < 4:
            ax.set_xticks([])         # Removes all tick marks
            ax.set_xlabel('')         # Removes the axis label
        ax.set_yticks([])         # Removes all tick marks
        ax.set_ylabel('')         # Removes the axis label

        ax = plt.subplot(4,4,4*plot_num+4)
        # Plotting the NARGP predictions with uncertainty estimates
        plt.plot(scaler.inverse_transform(Xtest)[:,0], kr_mean, color = 'blue', label = 'Kriging')
        plt.fill_between(scaler.inverse_transform(Xtest)[:,0], kr_mean-kr_conf, kr_mean + kr_conf, color = 'lightblue', alpha = 0.3)
        # Plotting the Fidelity-3 training data for comparison 
        inds = (scaler.inverse_transform(data_dict[3]['X'])[:,1] == test_temp) 
        Xsim, Ysim = data_dict[3]['X'][inds,:], data_dict[3]['Y'][inds]
        plt.scatter(scaler.inverse_transform(Xsim)[:,0], (Ysim), marker = '.', color = 'black', label = 'Lu 206-Step Mechanism')
        # Plotting the unseen high-fidelity testing data 
        plt.scatter(Xtrue[:,0], Ytrue, marker = '+', color = 'red', label = "High-Fidelity Testing Data")
        if plot_num + 1 == 1:
            plt.title("Kriging")
        if plot_num + 1 < 4:
            ax.set_xticks([])         # Removes all tick marks
            ax.set_xlabel('')         # Removes the axis label
        ax.set_yticks([])         # Removes all tick marks
        ax.set_ylabel('')         # Removes the axis label

        # # Plotting the Fidelity-3 training data for comparison 
        # inds = (scaler.inverse_transform(data_dict[3]['X'])[:,1] == test_temp) 
        # Xsim, Ysim = data_dict[3]['X'][inds,:], data_dict[3]['Y'][inds]
        # plt.scatter(scaler.inverse_transform(Xsim)[:,0], (Ysim), marker = '.', color = 'black', label = 'Lu 206-Step Mechanism')
        # # Plotting the unseen high-fidelity testing data 
        # plt.scatter(Xtrue[:,0], Ytrue, marker = '+', color = 'red', label = "High-Fidelity Testing Data")
        # Plot labeling 
        # plt.title("Predictions at %dK" % (test_temp))
        # if plot_num > 1:
        #     plt.xlabel("Equivalence Ratio, $\phi$")
        # if plot_num == 0 or plot_num == 2:
        #     plt.ylabel("Log Laminar Flame Speed - log(m/s)")
        # if plot_num == 0:
        #     plt.legend()

        # Fitting a degree-three polynomial through the testing points so we have a more dense error metric
        features = PolynomialFeatures(degree=3)
        lin_model = Ridge(alpha=1e-5)
        lin_model.fit(features.fit_transform(Xtrue[:,0].reshape(-1,1)), Ytrue)
        Yhat = lin_model.predict(features.transform(scaler.inverse_transform(Xtest)[:,0].reshape(-1,1))).ravel()

        lin_model.fit(features.fit_transform(scaler.inverse_transform(Xsim)[:,0].reshape(-1,1)), Ysim)
        Yhat_lu = lin_model.predict(features.transform(scaler.inverse_transform(Xtest)[:,0].reshape(-1,1))).ravel()

        print("Method (%sK)    RMSE         R^2      log MLL" % (test_temp))
        print("--------------------------------------------------")
        print("MAGPI:           %.3e &  %.4f &  %.4f \\\\" % (np.sqrt(np.mean(magpi_cov + (Yhat - magpi_mean)**2)), np.corrcoef(Yhat.ravel(), magpi_mean.ravel())[0,1]**2, -neg_mll(magpi.d[4]['model'], magpi.d[4]['model'].p)))
        print("Kennedy O'Hagan: %.3e &  %.4f &  %.4f \\\\" % (np.sqrt(np.mean(koh_cov + (Yhat - koh_mean)**2)), np.corrcoef(Yhat.ravel(), koh_mean.ravel())[0,1]**2, -delta_neg_mll(koh.d[4]['model'], koh.d[4]['model'].p)))
        print("NARGP:           %.3e &  %.4f &  %.4f \\\\" % (np.sqrt(np.mean(nargp_cov + (Yhat - nargp_mean)**2)), np.corrcoef(Yhat.ravel(), nargp_mean.ravel())[0,1]**2, -neg_mll(nargp.d[4]['model'], nargp.d[4]['model'].p)))
        print("Kriging:         %.3e &  %.4f &  %.4f \\\\" % (np.sqrt(np.mean(kr_cov + (Yhat - kr_mean)**2)), np.corrcoef(Yhat.ravel(), kr_mean.ravel())[0,1]**2, -neg_mll(kr, kr.p)))
    
    fig.supylabel("Log Laminar Flame Speed - log(m/s)", fontsize=16)
    fig.suptitle("Laminar Flame Speed Predictions at Various Temperatures", fontsize=18)
    fig.supxlabel("Equivalence Ratio, $\phi$", fontsize=16, ha='center', va = 'bottom', x=0.51, y = 0.08)
    all_handles = []
    all_labels = []
    # Loop through every axis in the flattened 4x4 array
    for ax in axs.flatten():
        handles, labels = ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)

    # Clean up duplicates (optional but often useful)
    # Use a dictionary to maintain order and uniqueness
    unique_handles_labels = dict(zip(all_labels, all_handles))
    final_handles = unique_handles_labels.values()
    final_labels = unique_handles_labels.keys()
    inds = [2,1,0,3,4,5]
    final_handles = [list(final_handles)[i] for i in inds]
    final_labels = [list(final_labels)[i] for i in inds]
    fig.tight_layout(rect=[0, 0.07, 1, 0.99])
    fig.legend(
        final_handles, 
        final_labels,
        loc = 'upper center',
        bbox_to_anchor = (0.5, 0.08),
        ncol = 3,
        fontsize = 12
    )

    plt.savefig("results/composite.png")


    # Plotting a three-dimensional scatterplot of the training and testing high-fidelity data
    fig = plt.figure(figsize=(9, 5), dpi=250)

    # Define the grid layout: 2 rows × 2 columns
    # 3D plot occupies subplots (1,3) — i.e., both rows of the first column
    ax3d = fig.add_subplot(2, 2, (1, 3), projection='3d')

    level = 4
    X_inv = scaler.inverse_transform(data_dict[level]['X'])
    Y_exp = jnp.exp(data_dict[level]['Y']).ravel()
    marker_size = 35

    # --- 3D Scatter Plot ---
    ax3d.scatter(X_inv[:, 0], X_inv[:, 1], Y_exp, c='black', marker='*', s = marker_size, label='Training Data')
    ax3d.scatter(Xvis[:, 0], Xvis[:, 1], jnp.exp(Yvis).ravel(), marker='+', c='red', label='Unseen Testing Data')

    ax3d.set_xlabel('Equivalence Ratio, $\phi$')
    ax3d.set_ylabel('Temperature, $T_0$')
    ax3d.set_zlabel('LFS (m/s)', rotation='vertical')
    ax3d.set_title('Laminar Flame Speed vs. \n Equivalence Ratio & Temperature')

    plt.subplots_adjust(wspace=0.00, hspace=0.55, right=0.95)
    box = ax3d.get_position()
    ax3d.set_position([box.x0 - 0.15, box.y0, box.width * 1.15, box.height])

    # --- 2D Marginal Plot 1 (Top Right: subplot 2) ---
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(X_inv[:, 0], Y_exp, c='black', marker='*', s = marker_size, label='Training Data', alpha=0.7)
    ax2.scatter(Xvis[:, 0], jnp.exp(Yvis).ravel(), c='red', marker='+', label='Testing Data')
    ax2.set_xlabel('Equivalence Ratio, $\phi$')
    ax2.set_ylabel('LFS (m/s)')
    ax2.set_title('Laminar Flame Speed vs. Equivalence Ratio')

    # --- 2D Marginal Plot 2 (Bottom Right: subplot 4) ---
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(X_inv[:, 1], Y_exp, c='black', marker='*', s = marker_size, label='Training Data', alpha = 0.7)
    ax4.scatter(Xvis[:, 1], jnp.exp(Yvis).ravel(), c='red', marker='+', label='Testing Data')
    ax4.set_xlabel('Temperature, $T_0$')
    ax4.set_ylabel('LFS (m/s)')
    ax4.set_title('Laminar Flame Speed vs. Temperature')
    ax4.legend(loc='upper left', fontsize=10)

    # plt.tight_layout()
    plt.savefig("results/LFSobjective.png")

    
