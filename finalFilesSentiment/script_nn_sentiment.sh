#!/bin/bash -l
# created: Nov 10, 2017 4:50 PM
# author: mushtaqm
#SBATCH -J sentimentAdnan
#SBATCH --constraint="snb|hsw"
#SBATCH -o output.out
#SBATCH -e error.err
#SBATCH -p serial
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-type=END
#SBATCH --mail-user=mushtaq@student.tut.fi

# commands to manage the batch script
#   submission command
#     sbatch [script-file]
#   status command
#     squeue -u mushtaqm
#   termination command
#     scancel [jobid]

# For more information
#   man sbatch
#   more examples in Taito guide in
#   http://research.csc.fi/taito-user-guide

# example run commands
srun python ./lstm.py

# This script will print some usage statistics to the
# end of file: output.out
# Use that to improve your resource request estimate
# on later jobs.
used_slurm_resources.bash
