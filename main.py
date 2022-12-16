from particleAI import ParticleAI
import numpy as np

#============================================
# SETTINGS

# DATA
path_MC = "/data/monte_carlo_data/" # Path to your Monte Carlo Files (Do not forget a slash at the end)
files_MC = ["data_etap_cut.txt", "data_eta_cut.txt", "data_pi0_cut.txt", "data_omega_cut.txt", "data_2pi0_cut.txt", "data_pi0eta_cut.txt"] # Only the first file may contain signal events. Add as many background files as you want.
ratio_MC = np.array([0.0031, 0.0793, 0.896, 0.0061, 0.0126, 0.0027]) # Ratio of your particles. This is very important to successfully train the NN.

path_real = "/data/real_data/" # Path to the real data (Do not forget a slash at the end)
files_real = ["data00.txt","data01.txt","data02.txt","data03.txt"] # All files of the real data


# NEURAL NET
neurons = [100,50] # List of neurons in the layers. If you add more numbers more layers will be added
epochs = 100 # Number of training epochs. 100 is usually good. If you've got datasizes greater than 10 Million you might increase this number.
batch = 0.005 # This effects the batchsize with batch * datasize = batchsize. Lower values lead to more corrections of the NN in a single epoch but slow down the training.
saperate_weights = True # If you've got timeweights in your real data, put them on the last column in the txt-files and set this to True.

# If you want to do a test run and don't care about accuracies. Just take epochs = 1 and batch = 0.1. This will heavily speed up your NN.



# CHOOSE THE MODE
# 0: Show your MC data to test your ratio_MC and get a fancy pic.
# 1: Train and test your Neural Net with the MC. You need this to get your model and its accuracy.
# 2: Same as 1 but without training. (saves time)
# 3: Use your trained Neural Net on the real data.

mode = 0


#================================================================
# MAIN 

if __name__ == "__main__":
  neuralnet = ParticleAI(path_MC, files_MC, ratio_MC, path_real, files_real, neurons=neurons, epochs=epochs, batch=batch, saperate_weights=saperate_weights)
  neuralnet.run(mode)
