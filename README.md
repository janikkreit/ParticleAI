# ParticleAI

This program was developed to enhance the signal-to-noise-ratio in particle physics analysis. 




# 1 Setup

In order to use this program you need to have Python 3 installed.
Furthermore some Python-Packages are required:
- numpy
- matplotlib
- seaborn
- tensorflow
- sklearn

 ROOT is commonly used in particle physics for data analysis so it is assumed to be installed too.


# 2 Preparation of data

The Simulation data as well as the real, measured data are stored in .root files. Python cannot read these files so you will convert the data to .txt files to use it.

The Neural Net need access to parameters of a single event. That means the text-file must contain your single reaction in rows and the parameter in columns. You can use any parameter you might think is important for the NN. Most important are energies and flight directions (i.e. polar and azimuthal angles) of the measured particles in your detector-system as well as beam energy. Further parameter can be added to improve the accuracy of the Network.

If you don't know how to create the .txt files you can use the convert_tree_to_txt.C program. For this to work a ROOT Tree is required in your .root files.
You will need to change some things before it will work for you.
- convert_tree_to_txt()
  - path to the root files and names of the included trees
  - the number of trees you are using
  - arguments in fill_files()
- make_files()
  - path where the text-files should be created
  - names of the columns in the text-file with suitable name for the Branches of the Tree
- fill_files()
  - arguments in the header
  - the list of trees t
  - variable names and branches of all the paramter you want to use in the NN
  - varibale names at the end for storing in the text-file

If you changed everything you can run it with root convert_tree_to_txt.C

The simulation files must saperated in signal oder background files or use one file for each particle in the intermediate state.
The real cannot and should not be splitted up.

# 3 Use the Neural Net

After creating the text-files open the particleAI.py file to set up some last things. 
