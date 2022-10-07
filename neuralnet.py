import matplotlib.pyplot as plt # Plots
import time # Runtime
import numpy as np # Data
import seaborn as sns # Confusionmatrix

# For neural net
#import tensorflow as tf
from keras.models import Sequential, load_model
#from keras.models import load_model
from keras.layers import Dense, Dropout, BatchNormalization, Softmax
from keras.callbacks import EarlyStopping, ModelCheckpoint

# For data selection, analization
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

t0 = time.time()

#=========================================
# FUNCTIONS
    
def get_size(path, files):
    size_list = np.zeros(len(files))
    for i in range(len(files)):
        count = 0
        for line in open(path+files[i]).readlines():
            count += 1
        size_list[i] = count
    return size_list
    
    
def show_raw_data(path, files, ratio):
    size = get_size(path, files)
    rows = (np.min(size/ratio)*ratio).astype(int)

    b = 100
    a = 0.5
    l = 3
    h = u"step"
    fig, ax = plt.subplots()

    print("\nloading MC data...")
    for i in range(len(files)):
        data = np.loadtxt(path+files[i], skiprows=1, max_rows=rows[i])
        im = invmass(data)
        ax.hist(im, bins=b, alpha=a, linewidth=l, histtype=h, label=f"data {i}")
    print("MC data has loaded successfully.")
    print(np.sum(rows), "Events used.\n")
        
    ax.set_title("invariant mass")
    ax.set_xlabel("$m_{\gamma \gamma}$ / MeV")
    ax.set_ylabel("counts")
    ax.set_yscale("log")
    ax.legend()

    fig.savefig("raw_data_mode-0.png")




def load_MC_data(path, files, ratio, test_size=0.3):
    print("\nloading MC data...")
    size = get_size(path, files)
    rows = (np.min(size/ratio)*ratio).astype(int)

	# load and merge input data
    x_data = np.loadtxt(path+files[0], skiprows=1, max_rows=rows[0])
    signal_len = len(x_data)
    for i in range(1,len(files)):
        x_data = np.concatenate((x_data, np.loadtxt(path+files[i], skiprows=1, max_rows=rows[i])), axis=0)

    # create output array where "1" stands for signal and "0" for backgroud
    y_data = np.zeros(len(x_data))
    y_data[:signal_len] = 1
    
    # split data to test and train arrays
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, shuffle=True)
    
    # normalize data
    scaler = StandardScaler()
    scaler.fit(x_data)  
    x_train_norm = scaler.transform(x_train)  
    x_test_norm = scaler.transform(x_test)
    
    print("MC data has loaded successfully.")
    print(np.sum(rows), "Events used.\n")
    return scaler, x_train_norm, x_train, x_test_norm, x_test, y_train, y_test



def load_real_data(path, files, scaler):
    print("\nloading real data...")

    # load and merge input data
    x_data = np.loadtxt(path+files[0], skiprows=1)
    if len(files) > 1:
        for i in range(1,len(files)):
            x_data = np.concatenate((x_data, np.loadtxt(path+files[i], skiprows=1)), axis=0)

    # normalize data
    x_data_norm = scaler.transform(x_data)  
    
    print("real data has loaded successfully.\n")
    return x_data_norm, x_data
    
    

def run_neuralnet(x_train, y_train, neurons=[100,50], epochs=100, batch=0.005, tol=1e-6, loss="LogCosh", optimizer="Adam"):
    batchsize = int(batch*len(x_train))

    # define neural network
    model = Sequential()
    for n in neurons:
        model.add(Dense(n, activation="relu"))#, input_dim=len(x_train[0])))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    
    # set stopping criteria
    es = EarlyStopping(monitor="val_loss", mode="min", min_delta=tol, verbose=0, patience=10)
    
    # train the model
    model.fit(x_train, y_train, epochs=epochs, verbose=2, validation_split = 0.1, batch_size=batchsize, shuffle=True, callbacks=[es])
    
    return model
    

    
    
#===========================================
# ANALYSATION

    
def invmass(data):
    E1 = data[:,2]
    E2 = data[:,3]
    theta1 = data[:,4]
    theta2 = data[:,5]
    phi1 = data[:,6]
    phi2 = data[:,7]
    
    cos_a = np.sin(theta1) *np.sin(theta2) *np.cos(phi1 - phi2) + np.cos(theta1) *np.cos(theta2)

    return np.sqrt(2 *E1 *E2 *(1 - cos_a))  
    
    

def calc_accuracy(model, x_test, y_test):
    y_pred = np.round(model.predict(x_test)).astype(int).flatten()
    acc = np.round(100*np.sum(y_test == y_pred)/len(y_test),2)

    print()
    print("Accuracy:", acc, "%")
    print()
    
    return y_pred
    

#==========================================
# PLOTS

def confusion_plot(test, pred):
    cf_matrix = confusion_matrix(test, pred)
    

    sum_1 = cf_matrix[0,0]+cf_matrix[0,1]
    sum_2 = cf_matrix[1,0]+cf_matrix[1,1]

    m1 = np.zeros(shape=(2,2))
    m1[0,0] = cf_matrix[0,0]/sum_1
    m1[0,1] = cf_matrix[0,1]/sum_1
    m1[1,0] = cf_matrix[1,0]/sum_2
    m1[1,1] = cf_matrix[1,1]/sum_2


    sum_1 = cf_matrix[0,0]+cf_matrix[1,0]
    sum_2 = cf_matrix[0,1]+cf_matrix[1,1]

    m2 = np.zeros(shape=(2,2))
    m2[0,0] = cf_matrix[0,0]/sum_1
    m2[1,0] = cf_matrix[1,0]/sum_1
    m2[0,1] = cf_matrix[0,1]/sum_2
    m2[1,1] = cf_matrix[1,1]/sum_2

    fig, axarr = plt.subplots(2, constrained_layout=True)
    for m, ax in zip([m1,m2], axarr.flatten()):
        sns.heatmap(m, annot=True, fmt='.2%', cmap='Blues', ax=ax)

        ax.set_xlabel("Predicted Events")
        ax.set_ylabel("Actual Events")
        ax.xaxis.set_ticklabels(["Background","Signal"])
        ax.yaxis.set_ticklabels(["Background","Signal"])
        
    acc = np.round(100*np.sum(test == pred)/len(test),2)
    fig.suptitle(f"total accurracy: {acc} %")

    fig.savefig("confusion_plot_mode-1-2.png", dpi=300)


  
def plot_invmass_MC(x_test, y_test, y_pred):	
    invmass_etap = invmass(x_test[y_test==1,:])
    invmass_bg = invmass(x_test[y_test==0,:])
    
    invmass_etap_pred = invmass(x_test[y_pred==1,:])
    invmass_bg_pred = invmass(x_test[y_pred==0,:])
    
    b = 100
    a = 0.5
    l = 3
    h = u"step"
    fig, ax = plt.subplots()
    
    ax.hist(invmass_etap, bins=b, alpha=a, linewidth=l, histtype=h, label="signal")
    ax.hist(invmass_bg, bins=b, alpha=a, linewidth=l, histtype=h, label="background")
    
    ax.hist(invmass_etap_pred, bins=b, alpha=a, linewidth=l, histtype=h, label="pred. signal")
    ax.hist(invmass_bg_pred, bins=b, alpha=a, linewidth=l, histtype=h, label="pred. background")
    
    
    ax.set_title("invariant mass")
    ax.set_xlabel("$m_{\gamma \gamma}$ / MeV")
    ax.set_ylabel("counts")
    ax.set_yscale("log")
    ax.legend()

    fig.savefig("MC_invariant_mass_mode-1-2.png", dpi=300)
    
    
def plot_invmass_real(x_test, y_pred):	  
    invmass_signal_pred = invmass(x_test[y_pred==1,:])
    invmass_bg_pred = invmass(x_test[y_pred==0,:])
    
    b = 100
    a = 0.5
    l = 3
    h = u"step"
    fig, ax = plt.subplots()
    
    ax.hist(invmass_signal_pred, bins=b, alpha=a, linewidth=l, histtype=h, label="pred. signal")
    ax.hist(invmass_bg_pred, bins=b, alpha=a, linewidth=l, histtype=h, label="pred. background")
    
    
    ax.set_title("invariant mass")
    ax.set_xlabel("$m_{\gamma \gamma}$ / MeV")
    ax.set_ylabel("counts")
    ax.set_yscale("log")
    ax.legend()

    fig.savefig("predicted_invariant_mass_mode-3.png", dpi=300)
    
    
    
    
    
#============================================
# SETTINGS

# DATA
path_MC = "/hiskp3/kreit/data/data_higher_energy/" # Path to your Monte Carlo Files (Do not forget a slash at the end)
files_MC = ["data_etap_cut.txt", "data_etap_cut.txt", "data_pi0_cut.txt", "data_omega_cut.txt", "data_2pi0_cut.txt", "data_pi0eta_cut.txt"] # Only the first file may contain signal events. Add as many background files as you want.
ratio_MC = np.array([0.0031, 0.0793, 0.896, 0.0061, 0.0126, 0.0027]) # Ratio of your particles. This is very important to successfully train the NN.

path_real = "/hiskp3/kreit/data/data_real/" # Path to the real data (Do not forget a slash at the end)
files_real = ["data00.txt","data01.txt","data02.txt","data03.txt"] # All files of the real data


# NEURALNET
neurons = [100,50] # List of neurons in the layers. If you add more numbers more layers will be added
epochs = 100 # Number of training epochs. 100 is usually good. If you've got datasizes greater than 10 Million you might increase this number.
batch = 0.005 # This effects the batchsize with batch * datasize = batchsize. Lower values lead to more corrections of the NN in a single epoch but slow down the training.

# If you want to do a test run and don't care about accuracies. Just take epochs = 1 and batch = 0.1. This will heavily speed up your NN.




# CHOOSE THE MODE
# 0: Show your MC data to test your ratio_MC and get a fancy pic.
# 1: Train and test your Neural Net with the MC. You need this to get your model and its accuracy.
# 2: Same as 0 but without training. (saves time)
# 3: Use your trained Neural Net on the real data.

mode = 1


#================================================================
# MAIN 

if mode == 0:
	show_raw_data(path_MC, files_MC, ratio_MC)

if mode == 1:
    _, xtr_n, xtr, xte_n, xte, ytr, yte = load_MC_data(path_MC, files_MC, ratio_MC)
    m = run_neuralnet(xtr_n, ytr, neurons=[100,50], epochs=100, batch=0.005)
    m.save("model.h5")
    yp = calc_accuracy(m, xte_n, yte)
    plot_invmass_MC(xte, yte, yp)
    confusion_plot(yte, yp)

	
elif mode == 2:
    _, xtr_n, xtr, xte_n, xte, ytr, yte = load_MC_data(path_MC, files_MC, ratio_MC)
    m = load_model("model.h5")
    yp = calc_accuracy(m, xte_n, yte)
    plot_invmass_MC(xte, yte, yp)
    confusion_plot(yte, yp)

elif mode == 3:
	scaler, xtr_n, xtr, xte_n, xte, ytr, yte = load_MC_data(path_MC, files_MC, ratio_MC)
	xte_n, xte = load_real_data(path_real, files_real, scaler)
	m = load_model("model.h5")
	yp = np.round(m.predict(xte_n)).astype(int)
	plot_invmass_real(xte, yp)
	

	
print(f"time spent: {np.round((time.time()-t0)/60,1)} min")
plt.show()