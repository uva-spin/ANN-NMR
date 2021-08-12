# qmeter_simulation

This is a repository for a simulation of the Liverpool Qmeter device. Its purpose is to train the user of a Liverpool Qmeter on how the system's output will respond to various inputs.

The original simulation was written in Mathcad, a proprietary software. This is an attempt to rewrite the simulation to be open-source and more portable by using Python in combination with CERN's PyROOT library, among other scientific libraries for Python.

The simulation has since been extended to be used in National Instruments LabVIEW for its GUI support.

A full report detailing the justifications and explanation of how this project was written can be found in report.pdf.

#Requirements
Python 3.6 or higher
numpy 9 or higher
scipy 
CERN PyROOT
Python standard library
National Instruments LabVIEW

#Python Instructions
The main application is main.py. Run this using

python main.py

in the command line.

#Labview Instructions

The main application (Qmeter simulation) is qmeter.vi. The deuteron lineshape generator is deuteron/deuteron_vi.vi.

#Folders
data: Experimental data which is used to build the simulation
deuteron: code for simulating the deuteron signal
mathcad: The Mathcad 15 documents which this simulation is based on.
old: Deprecated code.
pictures: images taken which are used in the report.

For more information visit twist.phys.virginia.edu.

2019, Colin Crovella
