#!/bin/bash

# make local and remote paths environment variables
# add data to a new folder so as not to overwrite existing data

LOCAL_DATA_PATH="$HOME/Documents/fem_fenics/local_post/data"
scp "z5165170@katana.restech.unsw.edu.au:fem_fenics/data/*" "${LOCAL_DATA_PATH}"