# TO-DO: 
#  - need an eval_points function

from sc_solver_2 import SC_Solver
# from plot_sol import plot_sol
from fenics import *
import welfords
import h5py, numpy as np
import random
import time
import sys
import pickle

# exec(open("test_sd.py").read())

# -----------------------------------------------------------------------------------------------
def generate_kappa(a):
    # generate random kappa: uniform distribution between [-a,a]
    # kappa identified with omega for simplicity (see 2.2)
    return (random.random()*2-1)*a
# -----------------------------------------------------------------------------------------------

# determine whether code is run on local or katana configuration; set up accordingly...
arg_num = len(sys.argv)
match arg_num:
    case 3: # use given parameter values
        eps = float(sys.argv[1])
        n = int(sys.argv[2])
    case _: # debug mode automatically executed when run on VSCode
        eps = 0.00
        n = 2

f = h5py.File('eval_points.h5','r') # (!) need to set working directory; os.chdir() could be useful?
eval_points = np.transpose(np.array(f['data']))
solver = SC_Solver(eval_points, 100)

# format: [agg_stress,agg_pressure,agg_rot]
# agg_stress (e.g.) = (count,mean,M2), see welfords.py
agg_sd = [(0,0,0),(0,0,0),(0,0,0)]

start = time.time()
for i in range(n):
    print(i)
    kappa = generate_kappa(1)
    # U = solver.run_sd(eps*0)
    U = solver.run_sd(eps*kappa)
    sol_list = solver.eval_sol(U)
    agg_sd = welfords.update_agg(agg_sd,sol_list)
    # print(agg_sd[1][1][100])
    # print(agg_sd[0][0])
    # solver.plot_sol(U)
time_sd = time.time() - start

save_file = '../data/eps_' + str(eps) + '.pkl'
with open(save_file,'wb') as f:
    pickle.dump([eps,n,eval_points,agg_sd,time_sd],f)
    f.close()