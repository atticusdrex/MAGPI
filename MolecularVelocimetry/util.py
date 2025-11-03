import sys
sys.path.append("..")   # add parent folder (project/) to Python path
from gplib.mf import *   # now absolute import works
from scipy.interpolate import griddata
from math import floor 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import Rbf, griddata
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

# Plot parameters 
plt.rcParams.update({
    "font.family": "georgia",
    'text.latex.preamble': r'\\usepackage{amsmath}',
    'mathtext.fontset': 'cm',
})

# Paths to velocity data 
filenames = [
    'cbfh_RANS_velocity.dat',
    'cbfh_LES_dx0p500mm_velocity.dat',
    'cbfh_LES_dx0p250mm_velocity.dat',
    'cbfh_LES_dx0p177mm_velocity.dat',
    'cbfh_LES_dx0p125mm_velocity.dat'
]

# Perform gridded interpolations for comparing levels of fidelity
def to_grid(Xtest, Ytest, N_partitions=500):
    grid_x, grid_y = np.meshgrid(
        np.linspace(Xtest[:,0].min(), Xtest[:,0].max(), N_partitions),
        np.linspace(Xtest[:,1].min(), Xtest[:,1].max(), N_partitions)
    )

    grid_z = griddata((Xtest[:,0], Xtest[:,1]), Ytest, (grid_x, grid_y), method='linear')

    return grid_x, grid_y, grid_z

# Function to read in the input files
def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            # Try to convert the line into 4 floats, skip if it fails
            parts = line.strip().split()
            if len(parts) == 4:
                try:
                    data.append([float(p) for p in parts])
                except ValueError:
                    continue  # skip lines that can't be parsed
    data = np.array(data)

    # Extract columns
    return data[:, :2], data[:, 2] , data[:, 3] 

def get_data_dict(target_qoi='U', grid_spacing=0.005):
    # initializing data dictionary 
    data_dict = {} 

    for level, file in enumerate(filenames):
        # Reading the data file 
        X, U, V = X, U, V = read_data('data/%s' % file)
        data_dict[level] = {
            'X':X, # XY coordinate as inputs to the model
            'U':U*0.01, # Horizontal component of velocity
            'V':V*0.01, # Vertical component of velocity
            'M':np.log(np.sqrt(U**2 + V**2) + 1e-5), # Computing the velocity magnitude
            'noise_var':1e-12 # Adding a small noise variance
        }

    # Aspect ratio for figures
    ratio = data_dict[4]['X'][:,0].max() / data_dict[4]['X'][:,1].max()
    ratio

    # Partitioning the x and y data 
    x_partitions = floor(0.04 / grid_spacing) + 1
    y_partitions = floor(0.0225 / grid_spacing) + 1
    x_vals = np.linspace(0.00, (x_partitions-1) * grid_spacing, x_partitions)
    y_vals = np.linspace(0.00, 0.0225, y_partitions)

    train_inds = []
    tol = 6e-5

    # Iterating through and choosing when X and Y are those values
    for x_val in x_vals:
        for y_val in y_vals:
            these_inds = np.where((np.abs(data_dict[4]['X'][:,0] - x_val) < tol) & (np.abs(data_dict[4]['X'][:,1] - y_val) < tol))
            train_inds += list(these_inds[0].ravel())

    # Selecting Xtrain and Ytrain
    Xtrain = data_dict[4]['X'][train_inds,:]
    Ytrain = data_dict[4][target_qoi][train_inds] 

    # Processing the data 
    scaler = StandardScaler()

    scaler = scaler.fit(data_dict[0]['X'])
    Xtrain = scaler.transform(Xtrain)

    QoIs = ['U', 'V', 'M']

    for level in data_dict.keys():
        # Centering and scaling X and Y coordinates 
        data_dict[level]['X'] = scaler.transform(data_dict[level]['X'])
        # Determining which QoI will be output for hyperkriging model
        for level in data_dict.keys():
            data_dict[level]['Y'] = np.copy(data_dict[level][target_qoi])
    
    return scaler, data_dict, ratio, (Xtrain, Ytrain), x_partitions

def generate_features(data_dict, Xtrain, Xtest, n_neighbors = 10, p=2):
    
    U_KNN_models = {} 

    # Creating features for high-fidelity regression
    train_features = np.copy(Xtrain)
    test_features = np.copy(Xtest)

    # Printing progress 
    print("Training KNN Models...")
    for level in tqdm(range(len(data_dict)-1)):
        # Declaring new K Nearest Neighbors Classifier
        model = KNeighborsRegressor(n_neighbors = n_neighbors, weights='distance')
        
        # # Fitting the model to the training data
        model.fit(data_dict[level]['X'], data_dict[level]['Y'])

        # # Storing the model in the dictionary
        U_KNN_models[level] = model

        # # Making train and test predictions 
        train_pred, test_pred = model.predict(Xtrain), model.predict(Xtest)

        # Appending new predictions onto the train and test features
        train_features = np.hstack((train_features, train_pred.reshape(-1,1)))
        test_features = np.hstack((test_features, test_pred.reshape(-1,1)))

    return train_features, test_features