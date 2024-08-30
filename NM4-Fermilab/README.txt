# Instructions

## **Creating Training/Testing Data**

You will require separate training and testing datasets in order to train your model and evaluate it, respectively. To do this, you should first create training data.

### **Training Data**

You must have large amounts of training data in order to ensure that a model has the "oppurtunity" to be properly trained. The best way to accomplish this is by running many many 'jobs' on Rivanna, UVA's supercomputer. To submit several jobs to be ran in-parallel with one another, you need a scipt (e.g., .py), a .slurm file, and a .sh file to run.

Provided in this directory are those required files: Create_Training_Data.py, Create_Data.slurm, and Big_Data.slurm

To run this in Rivanna, you (or hopefully should) have an account with Rivanna. Navigate to /project/ptgroup/your_directory and copy and paste this directory to there. If you dont have your own directory, create one! After creating a directory, navigate to it, open up the terminal from the right-hand corner, and run:

* ./Big_Data.sh <number of jobs> *

This will create a number of data events N, which is N = 1000 * (\# of jobs) which will be stored in a directory called "Training_Data", which will be created in the directory in which you are working.

## ** Training a Model ** 

