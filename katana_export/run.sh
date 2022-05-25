#!/bin/bash

# mk new data dir
# use TMPDIR node storage etc.

N=5000
# EPSILONS=[]
# for EPS in 1 0.7 0.5 0.2 0.1 0.07 0.05 0.02 0.01
# for EPS in 0.1 0.07 0.05 0.02 0.01 0.007 0.005 0.003 0.002 0.001 0.0
for EPS in 0.1 0.09 0.08 0.07 0.06 0.05 0.04 0.03 0.02 0.01 0.009 0.008 0.007 0.006 0.005 0.004 0.003 0.002 0.001 0.0
# for EPS in 0.02 0.01
# for EPS in 0.0
do
    qsub -v "eps=$EPS","n=$N" run_single.sh
done

echo "done!"