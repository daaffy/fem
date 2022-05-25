# TO-DO: 
#  - need an eval_points function

from sc_solver_2 import SC_Solver
# from plot_sol import plot_sol
from fenics import *
# import welfords
from aggs import *
import h5py, numpy as np
import random
import time
import sys
import pickle
from experiment import *
# import plot

# exec(open("test_run_statistics.py").read())
random.seed(20)
# -----------------------------------------------------------------------------------------------
def generate_kappa(a):
    # generate random kappa: uniform distribution between [-a,a]
    # kappa identified with omega for simplicity (see 2.2)
    return (random.random()*2-1)*a
# -----------------------------------------------------------------------------------------------

# determine whether code is run on local or katana configuration; set up accordingly...
arg_num = len(sys.argv)
is_local = True
match arg_num:
    case 3: # use given parameter values
        eps = float(sys.argv[1])
        n = int(sys.argv[2])
        is_local = False
    case _: # debug mode automatically executed when run on VSCode
        eps = 1
        n = 10

f = h5py.File('eval_points.h5','r') # (!) need to set working directory; os.chdir() could be useful?
eval_points = np.transpose(np.array(f['data']))
solver = SC_Solver(eval_points, 100)

# format: [agg_stress,agg_pressure,agg_rot]
# agg_stress (e.g.) = (count,mean,M2), see welfords.py
# agg_sd = [(0,0,0),(0,0,0),(0,0,0)]


e0 = experiment(eps,eval_points)
e_list = []

# agg_sd = aggs(3)
# sol_list = solver.test_sol(1)

# U = sol_list[0]
# plot.plot(eval_points,U)

# kappa_history = []
# start = time.time()
e0.begin()
for i in range(n):
    print(i)
    kappa = generate_kappa(1)
    # kappa_history.append(kappa)
    # U = solver.run_sd(eps*0)
    # U = solver.run_sd(eps*kappa)
    # sol_list = solver.eval_sol(U)
    # agg_sd = welfords.update_agg(agg_sd,sol_list)
    sol_list = solver.test_sol(kappa*eps)
    # agg_sd.update(sol_list)
    e0.update(kappa,sol_list)
    if (np.mod(i+1,2) == 0):
        e_i = e0
        e_i.end()
        e_list.append(e_i)
    # print(agg_sd[1][1][100])
    # print(agg_sd[0][0])
    # solver.plot_sol(U)
# time_sd = time.time() - start
# e0.end()

# --- temp ---
# import plot
# agg_1 = e0.agg_list[0]
# agg_1.finalise()
# U = agg_1.statistics[0]
# plot.plot(eval_points,U)

# --- export ---
# save_file = '../data/eps_' + str(eps) + '.pkl'
# e0.save(save_file)

# if (not is_local):
#     with open(save_file,'wb') as f:
#         pickle.dump([eps,n,eval_points,agg_sd,time_sd,kappa_history],f)
#         f.close()

if (not is_local):
    save_file = '../data/eps_' + str(eps) + '.pkl'
    with open(save_file,'wb') as f:
        pickle.dump(e_list,f)
        f.close()

# with open(save_file, 'rb') as f: # 'eps_...'
#     e_list = pickle.load(f)

# print(e_list[-1].agg_list[0].statistics)

# e2 = experiment.load(save_file)
# print(e2.t)

