#!/bin/bash

# make local and remote paths environment variables
# add data to a new folder so as not to overwrite existing data

NEWDIR=$1

LOCAL_DATA_PATH="$HOME/Documents/fem_fenics/local_post/data/"
cd $LOCAL_DATA_PATH
mkdir $NEWDIR
cd $NEWDIR
scp "z5165170@katana.restech.unsw.edu.au:fem_fenics/data/*" $(pwd)
