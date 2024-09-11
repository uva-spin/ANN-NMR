# Instructions

## **Creating Training/Testing Data**

You will require separate training and testing datasets in order to train your model and evaluate it, respectively. To do this, you should first create training data.

### **Training Data**

You must have large amounts of training data in order to ensure that a model has the "oppurtunity" to be properly trained. The best way to accomplish this is by running many many 'jobs' on Rivanna, UVA's supercomputer. To submit several jobs to be ran in-parallel with one another, you need a scipt (e.g., .py), a .slurm file, and a .sh file to run.

Provided in this directory are those required files: Create_Training_Data.py, Create_Data.slurm, and Big_Data.slurm

To run this in Rivanna, you (or hopefully should) have an account with Rivanna. Navigate to /project/ptgroup/your_directory and copy and paste this directory to there. If you dont have your own directory, create one! After creating a directory, navigate to it, open up the terminal from the right-hand corner, and run:

*./Big_Data.sh \<number of jobs\>*

This will create a number of data events N, which is N = 1000 * (\# of jobs) which will be stored in a directory called "Training_Data", which will be created in the directory in which you are working.

Afterwards, you will need to *merge* all of the files in */Training_Data* into a single file. To do this, just run

*sbatch Merge.slurm \<name of file you'd like to save to\>*

This will run the *Merge.py* script to automatically merge all of files in that directory into a single file.

To create Testing Data, data which with you can evaluate the performance of a model, all of the steps are completely similar, expcet this time, the data will be created with the script *Create_Training_Data.py*.

## **Training a Model** 

To train the model, you will need to either run the code locally (on your computer) or on Rivanna, depending on how good your setup is. For most people, it would be best to run the code on Rivanna.

To do so, navigate to your directory on Rivanna and open the terminal from the right-hand corner. Then, type

**sbatch Train_NN.slum**

This will run the script that will train the model. The script is already set up to automatically search for the training data that will be used to train the model. *Make sure* that you edit the name of the file to look for in *Training.py* to be the same as the sample data file you generated.  

## **Testing a Model**

After training a model, you will want to test how it will predict against the training data (data that it has never 'seem' before).To do this, you will need to use the script *Predict.py*. To submit a slurm job for this script, open the terminal and type:

*sbatch Predict.slurm \<Name of model\> \<training data\>* 


