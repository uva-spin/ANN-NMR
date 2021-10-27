# Integrated-Neural-Net

Important notes for using the integrated neural network:

  1. You must specify both an xml file and a txt file before the sim and neural net can run. A sample XML file that is trained on low levels of noise is included as 'Parameters.xml' in the Neural Net folder.
  2. You must specify an xml file again at the end of the simulation in order for it to end properly. I included this feature so that the user can save a new set of trained parameters without having to overwrite the original ones.
  3. Currently the only working optimizer is Adam. The other two optimizers that are listed (Adagrad and Adamax) required heavy use of Labviews confusing and inconsistent linear algebra functions. Through testing on other platforms I have found Adam to be the best optimizer anyways, so that is the only one I feel necessary to use in the sim.
  4. Importantly, I have left the other two optimization options in the program in case I can find a better way to include them in the future. Thus, IF YOU WANT TO TRAIN THE NETWORK YOU MUST SELECT ADAM AS THE OPTIMIZER. Turning on training with either of the other two optimizers will just leave the parameters unchanged.
  5. Lastly, It is important to let the Reimann sum of P approach the saturation polarization level before initiating training. Otherwise, with the way the program is written, the network will train towards incorrect values.
