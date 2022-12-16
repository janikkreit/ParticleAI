import time # Runtime
import matplotlib.pyplot as plt # Plots
import numpy as np # Data
import seaborn as sns # Confusionmatrix

# For neural net
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# For data selection, analization
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler



#=========================================
# LOAD DATA
    
class ParticleAI(object):
    def __init__(self, path_MC, files_MC, ratio_MC, path_real, files_real, neurons=[100,50], epochs=100, batch=0.005, saperate_weights=True):
        self.Path_MC = path_MC
        self.Files_MC = files_MC
        self.Ratio_MC = ratio_MC

        self.Path_real = path_real
        self.Files_real = files_real

        self.Neurons = neurons
        self.Epochs = epochs
        self.Batch = batch
        self.Saperate_weights = saperate_weights

    # get number of entries in a txt file
    def get_size(self, path, files):
        size_list = np.zeros(len(files))
        for i in range(len(files)):
            count = 0
            for line in open(path+files[i]).readlines():
                count += 1
            size_list[i] = count
        return size_list
        
    # plot the invariant mass of all particles 
    def show_raw_data(self):
        size = self.get_size(self.Path_MC, self.Files_MC)
        rows = (np.min(size/self.Ratio_MC)*self.Ratio_MC).astype(int)

        b = 100
        a = 0.5
        l = 3
        h = u"step"
        fig, ax = plt.subplots()

        print("\nloading MC data...")
        for i in range(len(self.Files_MC)):
            data = np.loadtxt(self.Path_MC+self.Files_MC[i], skiprows=1, max_rows=rows[i])
            im = self.invmass(data)
            ax.hist(im, bins=b, alpha=a, linewidth=l, histtype=h, label=f"data {i}")
        print("MC data has loaded successfully.")
        print(np.sum(rows), "Events used.\n")
            
        ax.set_title("invariant mass")
        ax.set_xlabel("$m_{\gamma \gamma}$ / MeV")
        ax.set_ylabel("counts")
        ax.set_yscale("log")
        ax.legend()

        fig.savefig("raw_data_mode-0.png")

        
    # load the monte carlo data and return arrays you can train/test the neural net
    def load_MC_data(self, test_size=0.3):
        print("\nloading MC data...")
        size = self.get_size(self.Path_MC, self.Files_MC)
        rows = (np.min(size/self.Ratio_MC)*self.Ratio_MC).astype(int)

        # load and merge input data
        x_data = np.loadtxt(self.Path_MC+self.Files_MC[0], skiprows=1, max_rows=rows[0])
        signal_len = len(x_data)
        for i in range(1,len(self.Files_MC)):
            x_data = np.concatenate((x_data, np.loadtxt(self.Path_MC+self.Files_MC[i], skiprows=1, max_rows=rows[i])), axis=0)

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


    # load real data and return arrays you can run the neural net with
    def load_real_data(self, scaler):
        print("\nloading real data...")

        # load and merge input data
        x_data = np.loadtxt(self.Path_real+self.Files_real[0], skiprows=1)
        if len(self.Files_real) > 1:
            for i in range(1,len(self.Files_real)):
                x_data = np.concatenate((x_data, np.loadtxt(self.Path_real+self.Files_real[i], skiprows=1)), axis=0)

        # saperate time-weigths from data
        if self.Saperate_weights == True:
            weights = x_data[:,-1]
            x_data = x_data[:,:-1]

        # normalize data
        x_data_norm = scaler.transform(x_data)  
        
        print("real data has loaded successfully.")
        print(len(x_data), "Events used.\n")
        return x_data_norm, x_data, weights
        

    #===========================================
    # NEURAL NET 

    # train the neural net
    def run_neuralnet(self, x_train, y_train, neurons=[100,50], epochs=100, batch=0.005, tol=1e-6, loss="LogCosh", optimizer="Adam"):
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

    # return the invariant mass of two photons
    def invmass(self, data):
        E1 = data[:,2]
        E2 = data[:,3]
        theta1 = data[:,4]
        theta2 = data[:,5]
        phi1 = data[:,6]
        phi2 = data[:,7]
        
        cos_a = np.sin(theta1) *np.sin(theta2) *np.cos(phi1 - phi2) + np.cos(theta1) *np.cos(theta2)

        return np.sqrt(2 *E1 *E2 *(1 - cos_a))  
        
    # calculate the accuracy of the trained neural net
    def calc_accuracy(self, model, x_test, y_test):
        y_pred = np.round(model.predict(x_test)).astype(int).flatten()
        acc = np.round(100*np.sum(y_test == y_pred)/len(y_test),2)

        print()
        print("Accuracy:", acc, "%")
        print()
        
        return y_pred
        


    #==========================================
    # PLOTS

    # plt a cufusion plot to get the accuracies of the neural net
    def confusion_plot(self, test, pred):
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

        fig, axarr = plt.subplots(1,2, constrained_layout=True)
        for m, ax in zip([m1,m2], axarr.flatten()):
            sns.heatmap(m, annot=True, fmt='.2%', cmap='Blues', ax=ax)

            ax.set_xlabel("Predicted Events")
            ax.set_ylabel("Actual Events")
            ax.xaxis.set_ticklabels(["Background","Signal"])
            ax.yaxis.set_ticklabels(["Background","Signal"])
            
        acc = np.round(100*np.sum(test == pred)/len(test),2)
        fig.suptitle(f"total accurracy: {acc} %")

        fig.savefig("confusion_plot_mode-1-2.png", dpi=300)


    # plot the invariant mass of two photons of the monte carlo
    def plot_invmass_MC(self, x_test, y_test, y_pred):	
        invmass_etap = self.invmass(x_test[y_test==1,:])
        invmass_bg = self.invmass(x_test[y_test==0,:])
        
        invmass_etap_pred = self.invmass(x_test[y_pred==1,:])
        invmass_bg_pred = self.invmass(x_test[y_pred==0,:])
        
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
        
        
    # plot the invariant mass of two photons of the real data
    def plot_invmass_real(self, x_test, y_pred, weights):	  
        invmass_signal_pred = self.invmass(x_test[y_pred==1,:])
        invmass_bg_pred = self.invmass(x_test[y_pred==0,:])
        
        b = 100
        a = 0.5
        l = 3
        h = u"step"
        fig, ax = plt.subplots()
        
        ax.hist(invmass_signal_pred, weights=weights[y_pred==1], bins=b, alpha=a, linewidth=l, histtype=h, label="pred. signal")
        ax.hist(invmass_bg_pred, weights=weights[y_pred==0], bins=b, alpha=a, linewidth=l, histtype=h, label="pred. background")
        
        ax.set_title("invariant mass")
        ax.set_xlabel("$m_{\gamma \gamma}$ / MeV")
        ax.set_ylabel("counts")
        ax.set_yscale("log")
        ax.legend()

        fig.savefig("predicted_invariant_mass_mode-3.png", dpi=300)


    # run defferent methods
    def run(self, mode):
        t0 = time.time()

        if mode == 0:
            self.show_raw_data()

        elif mode == 1:
            _, xtr_n, xtr, xte_n, xte, ytr, yte = self.load_MC_data()
            m = self.run_neuralnet(xtr_n, ytr, neurons=[100,50], epochs=100, batch=0.005)
            m.save("model.h5")
            yp = self.calc_accuracy(m, xte_n, yte)
            self.plot_invmass_MC(xte, yte, yp)
            self.confusion_plot(yte, yp)

            
        elif mode == 2:
            _, xtr_n, xtr, xte_n, xte, ytr, yte = self.load_MC_data()
            m = load_model("model.h5")
            yp = self.calc_accuracy(m, xte_n, yte)
            self.plot_invmass_MC(xte, yte, yp)
            self.confusion_plot(yte, yp)

        elif mode == 3:
            scaler, xtr_n, xtr, xte_n, xte, ytr, yte = self.load_MC_data()
            xte_n, xte, weights = self.load_real_data(scaler)
            m = load_model("model.h5")
            yp = np.round(m.predict(xte_n)).astype(int).flatten()
            self.plot_invmass_real(xte, yp, weights)

        else:
            print("Error: mode was not recognized.")

        print(f"time spent: {np.round((time.time()-t0)/60,1)} min")
        plt.show()
