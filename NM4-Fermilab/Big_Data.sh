#!/bin/bash

### To run this file, open up the terminal in Rivanna and 
### Type: "./jobscript.sh <number of jobs>"
### The amount of data events, in total, created = number of jobs * 1E3

dir_macros=$(dirname $(readlink -f $BASH_SOURCE))

njobs=$1

echo "njobs=$njobs"

for (( id=1; id<=$[$njobs]; id++ ))
do  
  echo "submitting job number = $id"
  sbatch Create_Sample_Data.slurm $id
  
done