#!/bin/bash

### qsub -v "eps=[float]","n=[int]" run_single.sh
### submit single run_statistics.py job for a given epsilon (1st argument)

#PBS -N fem_fenics
#PBS -l select=1:ncpus=1:mem=25gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -M j.hills@student.unsw.edu.au
#PBS -m ae

source ~/anaconda3/etc/profile.d/conda.sh # functions are not exported by default to be made available in subshells (https://github.com/conda/conda/issues/7980)
conda activate fenicsproject
cd ~/fem_fenics/src

# python katana_test.py $eps
python run_statistics.py $eps $n
