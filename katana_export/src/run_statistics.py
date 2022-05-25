from sc_solver_2 import SC_Solver
# from plot_sol import plot_sol
from fenics import *
import welfords
import h5py, numpy as np
import random
import time
import sys
import pickle
from aggs import *

# exec(open("test_sd.py").read())
random.seed(20)
# -----------------------------------------------------------------------------------------------
def generate_kappa(a):
    # generate random epsilon: uniform distribution between [-a,a]
    return (random.random()*2-1)*a
# -----------------------------------------------------------------------------------------------

arg_num = len(sys.argv)
print(sys.argv)
# print(arg_num)
match arg_num:
    case 3:
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
# agg_sd = [(0,0,0),(0,0,0),(0,0,0)]
agg_sd = aggs(3)

kappa_history = []
start = time.time()
for i in range(n):
    print(i)
    kappa = generate_kappa(1)
    # U = solver.run_sd(eps*0)
    U = solver.run_sd(eps*kappa)
    sol_list = solver.eval_sol(U)
    # agg_sd = welfords.update_agg(agg_sd,sol_list)
    agg_sd.update(sol_list)

    # print(agg_sd[1][1][100])
    # print(agg_sd[0][0])
    # solver.plot_sol(U)
time_sd = time.time() - start

save_file = '../data/eps_' + str(eps) + '.pkl'
with open(save_file,'wb') as f:
    pickle.dump([eps,n,eval_points,agg_sd,time_sd,kappa_history],f)
    f.close()