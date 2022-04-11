#!/bin/bash

N=10
# EPSILONS=[]
# for EPS in 1 0.7 0.5 0.2 0.1 0.07 0.05 0.02 0.01
for EPS in 0.02 0.01
do
    qsub -v "eps=$EPS","n=$N" run_single.sh
done

echo "done!"