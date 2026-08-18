from util import * 

plt.rcParams.update({
    "font.family": "serif",
    'font.serif': ["Times New Roman"],
    'text.latex.preamble': r'\\usepackage{amsmath}',
    'mathtext.fontset': 'stix',
})

def hf_plot():
    # Specifying target qoi and grid spacing 
    target_qoi, grid_spacing = 'U', 0.005

    # Loading data dictionary
    scaler, data_dict, ratio, train_tuple, x_partitions = get_data_dict(target_qoi=target_qoi, grid_spacing=grid_spacing) 

    # Extracting high-fidelity training data 
    Xtrain, Ytrain = train_tuple

    # Storing training and testing data
    Xtest, Ytest = data_dict[4]['X'], data_dict[4][target_qoi]
    data_dict[4]['X'], data_dict[4][target_qoi] = Xtrain, Ytrain

    # Visualizing High-Fidelity training data 
    with plt.rc_context({"font.size": 11}):
        plt.figure(figsize=(8.5, 2.5), dpi = 300)

        # Interpolating the data to a grid of points (500 partitions)
        X, Y, Z = to_grid(scaler.inverse_transform(Xtest), Ytest, ratio, X_partitions = 500)
        plt.pcolormesh(X, Y, Z, cmap = 'inferno')

        # Plotting the lower half of the mesh
        # temp_X = np.copy(scaler.inverse_transform(Xtest))
        # temp_X[:,1] = -temp_X[:,1]
        # temp_Y = np.copy(Ytest)
        # X, Y, Z = to_grid(temp_X, temp_Y, ratio, X_partitions = 500)
        # plt.pcolormesh(X, Y, Z, cmap = 'inferno')

        cbar = plt.colorbar(label = 'Horizontal Velocity (km/s)')
        cbar.ax.yaxis.label.set_rotation(-90)
        cbar.ax.yaxis.labelpad = 20
        cbar.ax.tick_params(labelsize=10)

        # plt.plot([0, 0.08], [0.0, 0.0], linestyle = 'dotted', linewidth =3.0, color = 'white', label = None, zorder=1)
        plt.scatter(scaler.inverse_transform(Xtrain)[:,0], scaler.inverse_transform(Xtrain)[:,1], s = 105.0, c = 'white', label = None, marker = 'P', zorder=2)
        plt.scatter(scaler.inverse_transform(Xtrain)[:,0], scaler.inverse_transform(Xtrain)[:,1], s = 75.0, c = 'black', label = "High-Fidelity Training Data", marker = '+', zorder=2)

        plt.xlim(-0.0, 0.08)
        plt.ylim(-0.0, 0.0225)

        plt.xlabel("X Coordinate (m)")
        plt.ylabel("Y Coordinate (m)")
        plt.title("125μm LES Horizontal Velocity Field Simulation", fontsize=11)
        plt.legend(fontsize=11, loc = 'upper right')
        plt.tight_layout()
        plt.savefig("results/sample_plot.png")

def hf_comparison_plot():
    # Specifying target qoi and grid spacing 
    target_qoi, grid_spacing = 'U', 0.005

    # Loading data dictionary
    scaler, data_dict, ratio, train_tuple, x_partitions = get_data_dict(target_qoi=target_qoi, grid_spacing=grid_spacing) 

    # Extracting high-fidelity training data 
    Xtrain, Ytrain = train_tuple

    # Storing training and testing data
    Xtest, Ytest = data_dict[4]['X'], data_dict[4][target_qoi]
    data_dict[4]['X'], data_dict[4][target_qoi] = Xtrain, Ytrain

    # Training low-fidelity surrogate models with KNN 
    train_features, test_features = generate_features(data_dict, Xtrain, Xtest, n_neighbors = 10)

    # Loading and making predictions with the Hyperkriging model 
    with open("models/magpi.pkl", "rb") as infile:
        magpi_model = pickle.load(infile)
    magpi_mean, magpi_var = magpi_model.predict(test_features, full_cov = False)

    # Visualizing High-Fidelity training data 
    with plt.rc_context({"font.size": 12}):
        fig = plt.figure(figsize=(10, 5.5), dpi = 400)

        # Making the norm 
        norm = TwoSlopeNorm(vmin=-0.2, vcenter=0.25, vmax=0.7)

        plt.subplot(2,1,1)
        # Interpolating the data to a grid of points (500 partitions)
        X, Y, Z = to_grid(scaler.inverse_transform(Xtest), Ytest, ratio, X_partitions = 500)
        plt.pcolormesh(X, Y, Z, cmap = 'inferno', norm = norm)

        

        cbar = plt.colorbar(label = 'Horizontal Velocity (km/s)')
        cbar.ax.yaxis.label.set_rotation(-90)
        cbar.ax.yaxis.labelpad = 20
        cbar.ax.tick_params(labelsize=10)

        # plt.plot([0, 0.08], [0.0, 0.0], linestyle = 'dotted', linewidth =3.0, color = 'white', label = None, zorder=1)
        plt.scatter(scaler.inverse_transform(Xtrain)[:,0], scaler.inverse_transform(Xtrain)[:,1], s = 105.0, c = 'white', label = None, marker = 'P', zorder=2)
        plt.scatter(scaler.inverse_transform(Xtrain)[:,0], scaler.inverse_transform(Xtrain)[:,1], s = 75.0, c = 'black', label = "Scarce Training Data", marker = '+', zorder=2)

        plt.xlim(-0.0, 0.08)
        plt.ylim(-0.0, 0.0225)
        plt.ylabel("Y Coordinate (m)")
        plt.title("High-Fidelity Flow Field Simulation", fontsize=11)
        plt.legend(fontsize=11, loc = 'upper right')
        plt.tight_layout()
        

        plt.subplot(2,1,2)
        # Hyperkriging Approximation
        X, Y, Z_hk = to_grid(scaler.inverse_transform(Xtest), magpi_mean, ratio, X_partitions = 500)
        plt.pcolormesh(X, Y, Z_hk, cmap = 'inferno', norm = norm)

        cbar = plt.colorbar(label = 'Horizontal Velocity (km/s)', norm=norm)
        cbar.ax.yaxis.label.set_rotation(-90)
        cbar.ax.yaxis.labelpad = 20
        cbar.ax.tick_params(labelsize=10)

        plt.xlim(-0.0, 0.08)
        plt.ylim(-0.0, 0.0225)

        plt.xlabel("X Coordinate (m)")
        plt.ylabel("Y Coordinate (m)")
        plt.title("Surrogate Model Prediction", fontsize=11)

        fig.subplots_adjust(left=0.10, right=1.0, bottom=0.10, top=0.90)
        fig.subplots_adjust(hspace=0.3)
        plt.savefig("results/hf_comparison_plot.png")

def comparison_plot():
    # Specifying target qoi and grid spacing 
    target_qoi, grid_spacing = 'U', 0.005

    # Loading data dictionary
    scaler, data_dict, ratio, train_tuple, x_partitions = get_data_dict(target_qoi=target_qoi, grid_spacing=grid_spacing) 

    # Extracting high-fidelity training data 
    Xtrain, Ytrain = train_tuple

    # Storing training and testing data
    Xtest, Ytest = data_dict[4]['X'], data_dict[4][target_qoi]
    data_dict[4]['X'], data_dict[4][target_qoi] = Xtrain, Ytrain

    # Training low-fidelity surrogate models with KNN 
    train_features, test_features = generate_features(data_dict, Xtrain, Xtest, n_neighbors = 10)

    # Getting the 177micrometer testing predictions 
    test_pred = test_features[:,-1]

    # Loading and making predictions with the Hyperkriging model 
    with open("models/magpi.pkl", "rb") as infile:
        magpi_model = pickle.load(infile)
    magpi_mean, magpi_var = magpi_model.predict(test_features, full_cov = False) 

    # Loading and making predictions with the Kriging model
    with open("models/kr.pkl", "rb") as infile:
        kr_model = pickle.load(infile)
    kr_mean, kr_var = kr_model.predict(Xtest, full_cov = False)

    # Making predictions using the kennedy o'hagan estimator
    with open("models/koh.pkl", "rb") as infile:
        delta=pickle.load(infile)
    delta_mean, delta_var = delta.predict(Xtest, full_cov = False)
    koh_mean = delta.p['rho'] * test_features[:,-1] + delta_mean
    koh_var = delta_var 

    # # Making predictions using the nargp estimator
    with open("models/nargp.pkl", "rb") as infile:
        nargp = pickle.load(infile)
    nargp_mean, nargp_var = nargp.predict(jnp.hstack((Xtest, test_features[:,-1].reshape(-1,1))), full_cov=False)

    # Specifying only the training data within the grid 
    grid_criterion = (Xtest[:,0] <= (x_partitions-1) * grid_spacing)


    # Creating the figure with a grid of subplots
    fig, axes = plt.subplots(3, 2, figsize=(8.5*1.5, 4.5*1.5), dpi=400)
    ax1, ax2, ax5, ax6, ax7, ax8 = axes.ravel()

    # # Making the 1,2 axis completely blank
    # ax2.axis("off")

    # # 125-resolution plot 
    X, Y, Z_125 = to_grid(scaler.inverse_transform(Xtest), Ytest, ratio, X_partitions = 500)
    # ax1.pcolormesh(X, Y, Z_125, cmap = 'inferno')
    # # ax1.plot([0, 0.08], [0.0, 0.0], linestyle = 'dotted', linewidth =1.0, color = 'black', label = 'Axis of Symmetry')
    # ax1.scatter(scaler.inverse_transform(Xtrain)[:,0], scaler.inverse_transform(Xtrain)[:,1], s = 105.0, c = 'white', label = None, marker = 'P')
    # ax1.scatter(scaler.inverse_transform(Xtrain)[:,0], scaler.inverse_transform(Xtrain)[:,1], s = 75.0, c = 'black', label = "High-Fidelity Training Data", marker = '+')
    # ax1.set_xlim(0.0, 0.08)
    # ax1.set_ylim(-0.000, 0.0225)
    # ax1.set_ylabel("Y Coordinate (m)")
    # ax1.set_title("125μm LES Simulation and Sparse Training Points")
    # ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    # ax1.legend(fontsize=15, loc = 'upper right')

    # 177 micrometer resolution plot
    X, Y, Z_177 = to_grid(scaler.inverse_transform(data_dict[3]['X']), data_dict[3][target_qoi], ratio, X_partitions = 500)
    im2 = ax1.pcolormesh(X, Y, Z_177, cmap = 'inferno')
    # ax3.plot([0, 0.08], [0.0, 0.0], linestyle = 'dotted', linewidth =1.0, color = 'black', label = 'Axis of Symmetry')
    ax1.set_title("177$\mu$m LES Simulation")
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax1.set_ylabel("Y Coordinate (m)")
    ax1.set_xlim(0.0, 0.08)
    ax1.set_ylim(-0.000, 0.0225)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # 177 vs. 125 error plot
    norm = TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=0.4)
    im4 = ax2.pcolormesh(X, Y, Z_177 - Z_125, cmap = 'RdBu', norm=norm)
    # ax4.plot([0, 0.08], [0.0, 0.0], linestyle = 'dotted', linewidth =1.0, color = 'black', label = 'Axis of Symmetry')
    ax2.set_title("177$\mu$m - 125$\mu$m Simulation")
    ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax2.set_xlim(0.0, 0.08)
    ax2.set_ylim(-0.000, 0.0225)

    # # 177 KNN approximation
    # X, Y, Z_KNN = to_grid(scaler.inverse_transform(Xtest), test_pred, ratio, X_partitions = 500)
    # ax3.pcolormesh(X, Y, Z_KNN, cmap = 'inferno')
    # # 3x5.plot([0, 0.08], [0.0, 0.0], linestyle = 'dotted', linewidth =1.0, color = 'black', label = 'Axis of Symmetry')
    # ax3.set_title("KNN Approximation of 177$\mu$m Simulation")
    # ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # ax3.set_ylabel("Y Coordinate (m)")
    # ax3.set_xlim(0.0, 0.08)
    # ax3.set_ylim(-0.000, 0.0225)
    # ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax3.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # # 177 vs. 177 KNN error plot
    # ax4.pcolormesh(X, Y, Z_KNN - Z_177, cmap = 'RdBu', norm=norm)
    # # 4x6.plot([0, 0.08], [0.0, 0.0], linestyle = 'dotted', linewidth =1.0, color = 'black', label = 'Axis of Symmetry')
    # ax4.set_title("KNN Prediction - 177$\mu$m Simulation")
    # ax4.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    # ax4.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # ax4.set_xlim(0.0, 0.08)
    # ax4.set_ylim(-0.000, 0.0225)

    # Hyperkriging Approximation
    X, Y, Z_hk = to_grid(scaler.inverse_transform(Xtest), magpi_mean, ratio, X_partitions = 500)
    ax5.pcolormesh(X, Y, Z_hk, cmap = 'inferno')
    # 5x7.plot([0, 0.08], [0.0, 0.0], linestyle = 'dotted', linewidth =1.0, color = 'black', label = 'Axis of Symmetry')
    ax5.set_title("Proposed Method Approximation of 125$\mu$m Simulation")
    ax5.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax5.set_ylabel("Y Coordinate (m)")
    ax5.set_xlim(0.0, 0.08)
    ax5.set_ylim(-0.000, 0.0225)
    ax5.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax5.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # Hyperkriging vs. 125 micrometer error plot
    ax6.pcolormesh(X, Y, Z_hk - Z_125, cmap = 'RdBu', norm=norm)
    ax6.plot([0, 0.08], [0.0, 0.0], linestyle = 'dotted', linewidth =1.0, color = 'black', label = 'Axis of Symmetry')
    ax6.set_title("Proposed Method Prediction - 125$\mu$m Simulation")
    ax6.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax6.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax6.set_xlim(0.0, 0.08)
    ax6.set_ylim(-0.000, 0.0225)

    # # Kennedy O'Hagan Approximation
    # X, Y, Z_koh = to_grid(scaler.inverse_transform(Xtest), koh_mean, ratio, X_partitions = 500)
    # ax7.pcolormesh(X, Y, Z_koh, cmap = 'inferno')
    # # 7x7.plot([0, 0.08], [0.0, 0.0], linestyle = 'dotted', linewidth =1.0, color = 'black', label = 'Axis of Symmetry')
    # ax7.set_title("Kennedy O'Hagan Approximation of 125$\mu$m Simulation")
    # ax7.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # ax7.set_ylabel("Y Coordinate (m)")
    # ax7.set_xlim(0.0, 0.08)
    # ax7.set_ylim(-0.000, 0.0225)
    # ax7.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax7.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # # Kennedy O'Hagan vs. 125 micrometer error plot
    # ax8.pcolormesh(X, Y, Z_koh - Z_125, cmap = 'RdBu', norm=norm)
    # # ax8.plot([0, 0.08], [0.0, 0.0], linestyle = 'dotted', linewidth =1.0, color = 'black', label = 'Axis of Symmetry')
    # ax8.set_title("Kennedy O'Hagan Prediction - 125$\mu$m Simulation")
    # ax8.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # ax8.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    # ax8.set_xlim(0.0, 0.08)
    # ax8.set_ylim(-0.000, 0.0225)

    # # NARGP Approximation
    # X, Y, Z_nargp = to_grid(scaler.inverse_transform(Xtest), nargp_mean, ratio, X_partitions = 500)
    # ax9.pcolormesh(X, Y, Z_nargp, cmap = 'inferno')
    # # 9x7.plot([0, 0.08], [0.0, 0.0], linestyle = 'dotted', linewidth =1.0, color = 'black', label = 'Axis of Symmetry')
    # ax9.set_title("NARGP Approximation of 125$\mu$m Simulation")
    # ax9.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # ax9.set_ylabel("Y Coordinate (m)")
    # ax9.set_xlim(0.0, 0.08)
    # ax9.set_ylim(-0.000, 0.0225)
    # ax9.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax9.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # # NARGP vs. 125 micrometer error plot
    # ax10.pcolormesh(X, Y, Z_nargp - Z_125, cmap = 'RdBu', norm=norm)
    # # ax10.plot([0, 0.08], [0.0, 0.0], linestyle = 'dotted', linewidth =1.0, color = 'black', label = 'Axis of Symmetry')
    # ax10.set_title("NARGP Prediction - 125$\mu$m Simulation")
    # ax10.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # ax10.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    # ax10.set_xlim(0.0, 0.08)
    # ax10.set_ylim(-0.000, 0.0225)

    # Kriging Approximation
    X, Y, Z_kr = to_grid(scaler.inverse_transform(Xtest), kr_mean, ratio, X_partitions = 500)
    ax7.pcolormesh(X, Y, Z_kr, cmap = 'inferno')
    # 7x7.plot([0, 0.08], [0.0, 0.0], linestyle = 'dotted', linewidth =1.0, color = 'black', label = 'Axis of Symmetry')
    ax7.set_title("Kriging Approximation of 125$\mu$m Simulation")
    ax7.set_xlabel("X Coordinate (m)")
    ax7.set_ylabel("Y Coordinate (m)")
    ax7.set_xlim(0.0, 0.08)
    ax7.set_ylim(-0.000, 0.0225)
    ax7.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax7.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # Kriging vs. 125 micrometer error plot
    ax8.pcolormesh(X, Y, Z_kr - Z_125, cmap = 'RdBu', norm=norm)
    # 812.plot([0, 0.08], [0.0, 0.0], linestyle = 'dotted', linewidth =1.0, color = 'black', label = 'Axis of Symmetry')
    ax8.set_title("Kriging Prediction - 125$\mu$m Simulation")
    ax8.set_xlabel("X Coordinate (m)")
    ax8.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax8.set_xlim(0.0, 0.08)
    ax8.set_ylim(-0.000, 0.0225)

    # Shared colorbar for left plots 
    cbar = fig.colorbar(im2, ax=[ax1, ax5, ax7], orientation='vertical', pad = 0.2, label = "Horizontal Velocity (km / s)")
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.label.set_rotation(-90)
    cbar.ax.yaxis.label.set_fontsize(15)
    cbar.ax.yaxis.labelpad = 20 
    pos = cbar.ax.get_position()  # get current position [x0, y0, width, height]
    cbar.ax.set_position([pos.x0 + 0.01, pos.y0, pos.width*0.5, pos.height])

    # Shared colorbar for right plots 
    cbar_right = fig.colorbar(im4, ax=[ax2, ax6, ax8], orientation='vertical', pad = 0.4, label = "Error (km / s)")
    cbar_right.ax.tick_params(labelsize=10)
    cbar_right.ax.yaxis.label.set_rotation(-90)
    cbar_right.ax.yaxis.label.set_fontsize(15)
    cbar_right.ax.yaxis.labelpad = 20 
    pos = cbar_right.ax.get_position()  # get current position [x0, y0, width, height]
    cbar_right.ax.set_position([pos.x0 + 0.07, pos.y0, pos.width*0.5, pos.height])

    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.06, top=0.97)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig("results/comparison_plot.png")

    # Computing performance metrics 
    magpi_rmse = jnp.sqrt(jnp.mean(magpi_var[grid_criterion]) + jnp.mean((magpi_mean[grid_criterion] - Ytest[grid_criterion])**2))
    koh_rmse = jnp.sqrt(jnp.mean(koh_var[grid_criterion] + (koh_mean[grid_criterion] - Ytest[grid_criterion])**2))
    nargp_rmse = jnp.sqrt(jnp.mean(nargp_var[grid_criterion] + (nargp_mean[grid_criterion] - Ytest[grid_criterion])**2))
    kr_rmse = jnp.sqrt(jnp.mean(kr_var[grid_criterion] + (kr_mean[grid_criterion] - Ytest[grid_criterion])**2))


    print("Method                     Grid RMSE      Grid R^2   Log ML")
    print("---------------------------------------------------------------------------")
    print("Proposed Method        &  %.4e  &  %.4f  &  %.4f \\\\" % (magpi_rmse, np.corrcoef(Ytest[grid_criterion], magpi_mean[grid_criterion])[0,1]**2, -neg_mll(magpi_model, magpi_model.p)))
    print("Kennedy OH             &  %.4e  &  %.4f  &  %.4f \\\\" % (koh_rmse, np.corrcoef(Ytest[grid_criterion], koh_mean[grid_criterion])[0,1]**2, -delta_neg_mll(delta, delta.p)))
    print("NARGP                  &  %.4e  &  %.4f  &  %.4f \\\\" % (nargp_rmse, np.corrcoef(Ytest[grid_criterion], nargp_mean[grid_criterion])[0,1]**2, -neg_mll(nargp, nargp.p)))
    print("Kriging                &  %.4e  &  %.4f  &  %.4f \\\\" % (kr_rmse, np.corrcoef(Ytest[grid_criterion], kr_mean[grid_criterion])[0,1]**2, -neg_mll(kr_model, kr_model.p)))
    print("177$\\mu$m             &  %.4e  &  %.4f  &  -- \\\\ " %  (jnp.sqrt(MSE(Ytest[grid_criterion], test_features[grid_criterion, 5])), np.corrcoef(Ytest[grid_criterion], test_features[grid_criterion, 5])[0,1]**2))
    print("250$\\mu$m             &  %.4e  &  %.4f  &  -- \\\\ " %  (jnp.sqrt(MSE(Ytest[grid_criterion], test_features[grid_criterion, 4])), np.corrcoef(Ytest[grid_criterion], test_features[grid_criterion, 4])[0,1]**2))
    print("500$\\mu$m             &  %.4e  &  %.4f  &  -- \\\\ " %  (jnp.sqrt(MSE(Ytest[grid_criterion], test_features[grid_criterion, 3])), np.corrcoef(Ytest[grid_criterion], test_features[grid_criterion, 3])[0,1]**2))
    print("RANS$\\mu$m            &  %.4e  &  %.4f  &  -- \\\\  \\hline" %  (jnp.sqrt(MSE(Ytest[grid_criterion], test_features[grid_criterion, 2])), np.corrcoef(Ytest[grid_criterion], test_features[grid_criterion, 2])[0,1]**2))

def validation_plot():
    # Specifying target qoi and grid spacing 
    target_qoi, grid_spacing = 'U', 0.005

    # Loading data dictionary
    scaler, data_dict, ratio, train_tuple, x_partitions = get_data_dict(target_qoi=target_qoi, grid_spacing=grid_spacing) 

    # Extracting high-fidelity training data 
    Xtrain, Ytrain = train_tuple

    # Storing training and testing data
    Xtest, Ytest = data_dict[4]['X'], data_dict[4][target_qoi]
    data_dict[4]['X'], data_dict[4][target_qoi] = Xtrain, Ytrain

    # Training low-fidelity surrogate models with KNN 
    train_features, test_features = generate_features(data_dict, Xtrain, Xtest, n_neighbors = 10)

    # Loading and making predictions with the Hyperkriging model 
    with open("models/magpi.pkl", "rb") as infile:
        magpi_model = pickle.load(infile)
    magpi_mean, magpi_var = magpi_model.predict(test_features, full_cov = False) 

    # Specifying only the training data within the grid 
    grid_criterion = (Xtest[:,0] <= (x_partitions-1) * grid_spacing)


    # Creating the figure with a grid of subplots
    fig, axes = plt.subplots(3,1, figsize=(8.5*1.5, 7*1.5), dpi=300)
    ax1, ax2, ax3 = axes.ravel()

    # # 125-resolution plot 
    X, Y, Z_125 = to_grid(scaler.inverse_transform(Xtest), Ytest, ratio, X_partitions = 500)

    # 177 vs. 125 error plot
    norm = TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2)

    # Hyperkriging Approximation
    X, Y, Z_hk = to_grid(scaler.inverse_transform(Xtest), magpi_mean, ratio, X_partitions = 500)
    _, _, Z_hk_var = to_grid(scaler.inverse_transform(Xtest), magpi_var, ratio, X_partitions = 500)
    # Hyperkriging vs. 125 micrometer error plot
    im1 = ax1.pcolormesh(X, Y, (Z_hk - Z_125), cmap = 'RdBu', norm=norm)
    ax1.plot([0, 0.08], [0.0, 0.0], linestyle = 'dotted', linewidth =1.0, color = 'black', label = 'Axis of Symmetry')
    ax1.set_title("Error between Proposed Method and High-Fidelity (125$\mu$m)", fontsize=15)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax1.set_ylabel("Y Coordinate (m)")
    ax1.set_xlim(0.0, 0.08)
    ax1.set_ylim(-0.000, 0.0225)
    cbar_left = fig.colorbar(im1, ax=[ax1], orientation='vertical', pad = 0.2, label = "Model Error (km / s)")
    cbar_left.ax.tick_params(labelsize=10)
    cbar_left.ax.yaxis.label.set_rotation(-90)
    cbar_left.ax.yaxis.label.set_fontsize(15)
    cbar_left.ax.yaxis.labelpad = 20
    pos = cbar_left.ax.get_position()  # get current position [x0, y0, width, height]
    cbar_left.ax.set_position([pos.x0 + 0.13, pos.y0+0.04, pos.width*0.5, pos.height])

    # Hyperkriging vs. 125 micrometer error plot
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.03, vmax=0.06)
    im2 = ax2.pcolormesh(X, Y, jnp.sqrt(Z_hk_var), cmap = 'Oranges', norm = norm)
    ax2.plot([0, 0.08], [0.0, 0.0], linestyle = 'dotted', linewidth =1.0, color = 'black', label = 'Axis of Symmetry')
    ax2.set_title("Predictive Uncertainty of Proposed Method", fontsize=15)
    ax2.set_xlabel("X Coordinate (m)")
    ax2.set_ylabel("Y Coordinate (m)")
    ax2.set_xlim(0.0, 0.08)
    ax2.set_ylim(-0.000, 0.0225)

    cbar_right = fig.colorbar(im2, ax=[ax2], orientation='vertical', pad = 0.2, label = "Model Uncertainty (km / s)")
    cbar_right.ax.tick_params(labelsize=10)
    cbar_right.ax.yaxis.label.set_rotation(-90)
    cbar_right.ax.yaxis.label.set_fontsize(15)
    cbar_right.ax.yaxis.labelpad = 20 
    pos = cbar_right.ax.get_position()  # get current position [x0, y0, width, height]
    cbar_right.ax.set_position([pos.x0 + 0.13, pos.y0-0.0, pos.width*0.5, pos.height])

    # Confidence Interval Plot
    from matplotlib.colors import ListedColormap, BoundaryNorm 
    cmap = ListedColormap(['white', 'green'])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    Z_correct = np.zeros_like(Z_hk)
    Z_hk_conf = jnp.sqrt(Z_hk_var)
    Z_correct[(Z_125 >= Z_hk - 2.58 * Z_hk_conf) & (Z_125 <= Z_hk + 2.58 * Z_hk_conf)] = 1
    print(jnp.mean(Z_correct))
    im3 = ax3.pcolormesh(X, Y, Z_correct, cmap = cmap, norm = norm)
    ax3.plot([0, 0.08], [0.0, 0.0], linestyle = 'dotted', linewidth =1.0, color = 'black', label = 'Axis of Symmetry')
    ax3.set_title("True Flow Field within 99% Confidence Interval", fontsize=15)
    ax3.set_xlabel("X Coordinate (m)")
    ax3.set_ylabel("Y Coordinate (m)")
    ax3.set_xlim(0.0, 0.08)
    ax3.set_ylim(-0.000, 0.0225)

    # Create colorbar
    cbar = fig.colorbar(im3, ax=[ax3], ticks=[0, 1], pad = 0.2, label = "Model Uncertainty")
    cbar.ax.set_yticklabels(['Incorrect', 'Correct'], fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.label.set_rotation(-90)
    cbar.ax.yaxis.label.set_fontsize(15)
    cbar.ax.yaxis.labelpad = 20 
    pos = cbar.ax.get_position()  # get current position [x0, y0, width, height]
    cbar.ax.set_position([pos.x0 + 0.13, pos.y0-0.04, pos.width*0.5, pos.height])

    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.1, top=0.95)
    fig.subplots_adjust(wspace=0.3, hspace=0.15)
    plt.savefig("results/validation_plot.png")

    # Computing the correlation between the error and the predictive uncertainty
    print("Error vs. Uncertainty Correlation")
    print(jnp.corrcoef(jnp.sqrt(Z_hk_var).ravel(), (jnp.abs((Z_hk - Z_125))).ravel())[0,1])


if __name__ == "__main__":
    comparison_plot()
    # hf_plot() 
    # validation_plot()
    # hf_comparison_plot()