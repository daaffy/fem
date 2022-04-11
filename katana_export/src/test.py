import random
import welfords
import h5py, numpy as np
import pickle
import sys
import time
from fenics import *
from multiphenics import *
from sc_solver import SC_Solver

# FUNCTIONS ----------------------------------------------------------------------------------------------

def generate_kappa(a):
    # generate random epsilon: uniform distribution between [-a,a]
    return (random.random()*2-1)*a

def update_agg(agg_list,sol_list):
    for i in range(len(agg_list)):
        agg_list[i] = welfords.update(agg_list[i],sol_list[i])
    return agg_list
# --------------------------------------------------------------------------------------------------------

# eps = float(sys.argv[1])
eps = 0.01

solver = SC_Solver(10)

f = h5py.File('eval_points.h5','r') # (!) need to set working directory; os.chdir() could be useful?
eval_points = np.transpose(np.array(f['data']))
solver.eval_at(eval_points) # solver returns values evaluated at these points
solver.set_domain("SQUARE")

n = 10
# format: [agg_stress,agg_pressure,agg_rot]

# kap_test = generate_kappa(1)
kap_test = 0

sol_0 = solver.run_sd(0,0)

# FIXED DOMAIN STATISTICS
start = time.time()
V = solver.load_fd()
U_fd_agg = [(0,BlockFunction(V),BlockFunction(V))]
print(U_fd_agg[0])

for i in range(0,n): # NOTe UPDATE RUN_FD TO INCORPORATE SPEED UP i.e., ONLY ASSEMBLE a ONCE ETC
    print(i)
    # sol_list = solver.run_fd(1)
    U_fd = solver.run_fd(1)
    
    if i == 0:
        # U_fd_agg = [(0,U_fd,U_fd)]
        U_fd_agg = [(0,U_fd,0*U_fd)]
    else:
        # U_fd_agg = update_agg(agg_fd,U_fd)
        U_fd_agg = U_fd + U_fd_agg

    # agg_fd = update_agg(agg_fd,sol_list)
    # print(sol_list[1])
time_fd = time.time() - start

# solver.plot_sol(3)
print(time_fd)
