# ParticleAI

This program was developed to enhance the signal-to-noise-ratio in particle physics analysis.
You can use [Bachelorthesis_Janik_Kreit.pdf](Bachelorthesis_Janik_Kreit.pdf) as a further documentation of my work.

In an applied experiment at the [ELSA](https://www-elsa.physik.uni-bonn.de/) (ELectron Stretcher
Accelerator) and the [CBELSA/TAPS](https://www.cb.uni-bonn.de/) at the university in Bonn accelerated electrons (up to an energy of 3 GeV) were used to produce photons via bremsstrahung. These photons collide with a target existing of hydrogen.
A reaction was invastigated where photons and protons produce the η′ meson. That means: ɣp -> pη′.
To extract events in which an η′ meson was created only ɣɣp final states were used. (An η′ meson decays with 2.3% to two photons.)

## Problem
By collinding photons and protons not only η′ are created. There are also 2π0, η and ω mesons which overlay the wanted reaction.
These mesons can decay in two or more photons and therefore appear in this analysis. If an meson decays to e.g. 4 photons it is possible that 2 photons do not hit any detector so the event looks like a two photon event.

## Solution
A neural network which knows reaction paramters like energy, azimuthal and polar angle of the proton and the two photons in the final state can than decide whether an η′ meson or some other meson was created. Monte Carlo simulations were used to train the neural net.

## Result
As a result of this type of analysis the neural net was able to detect 74% of all η′. This doen't seem much in the first place but with classical extraction methods there remain as much η′ mesons as other mesons. A signal-to-noise-ratio (SNR) can be determined by deviding the number of all η′ mesons by the number of all other (noise) mesons (2π0, η, ω). This leads to a much higher SNR using the neural network.

|| Classic | Neural Net |
| --- | --- | --- |
| SNR | 1.34 | 2.90 |



# Quick Start Guide
## 1 Setup

In order to use this program you need to have Python 3 installed.
Furthermore some Python-Packages are required:
- numpy
- matplotlib
- seaborn
- tensorflow
- sklearn

 ROOT is commonly used in particle physics for data analysis so it is assumed to be installed too.


## 2 Preparation of data

The Simulation data as well as the real, measured data are stored in .root files. Python cannot read these files so you will convert the data to .txt files to use it.

The neural net need access to parameters of a single event. That means the text-file must contain your single reaction in rows and the parameter in columns. You can use any parameter you might think is important for the network. Most important are energies and flight directions (i.e. polar and azimuthal angles) of the measured particles in your detector-system as well as beam energy. Further parameters can be added to improve the accuracy of the network.

If you don't know how to create the .txt files you can use the [convert_tree_to_txt.C](convert_tree_to_txt.C) program. For this to work a ROOT Tree is required in your .root files.
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
  - variable names and branches of all the paramter you want to use in the neural network
  - varibale names at the end for storing in the text-file

If you changed everything you can run it with `root convert_tree_to_txt.C` .

The simulation files must saperated in signal and background files; or use one file for each particle in the intermediate state.
The real data of the experiment cannot and should not be splitted up.

## 3 Use the Neural Net

After creating the text-files open the [main.py](main.py) file to set up some last things.
- specify the *path* and *files* of your data (.txt) files
- *ratio_MC* defines how often a specific event occurs relative to others. This is **very important** for a successful training of the neural net. See [Bachelorthesis_Janik_Kreit.pdf](Bachelorthesis_Janik_Kreit.pdf) Chapter 5.3.1 (Table 5.2) for more information.
- the settings for the neural net can changed for further improvements
- choose different modes (0 - 3) to show your data, train and test the neural net with the Monte Carlo, and use the neural net for the real data
- run [main.py](main.py)

**Note**
> The method *invmass(self, data)* in [particleAI.py](particleAI.py) requires two particles with parameters (e.g. energy) at the right columns in *data*. If you've got more particles in your analysis you need to edit this function.
