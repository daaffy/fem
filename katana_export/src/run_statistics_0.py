import random
import welfords
import h5py, numpy as np
import pickle
import sys
import time
import matplotlib.pyplot as plt
# from fenics import *
# from multiphenics import *
from sc_solver import SC_Solver

# FUNCTIONS ----------------------------------------------------------------------------------------------

def generate_kappa(a):
    # generate random epsilon: uniform distribution between [-a,a]
    return (random.random()*2-1)*a

def update_agg(agg_list,sol_list):
    for i in range(len(agg_list)):
        agg_list[i] = welfords.update(agg_list[i],sol_list[i])
    return agg_list

def finalize(agg_list):
    for i in range(len(agg_list)):
        agg_list[i] = welfords.finalize(agg_list[i])
    return agg_list

# --------------------------------------------------------------------------------------------------------

arg_num = len(sys.argv)
# print(sys.argv)
# print(arg_num)
match arg_num:
    case 3:
        eps = float(sys.argv[1])
        n = int(sys.argv[2])
    case _: # debug mode automatically executed when run on VSCode
        eps = 0.01
        n = 1

print(eps)
print(n)

solver = SC_Solver(60)

f = h5py.File('eval_points.h5','r') # (!) need to set working directory; os.chdir() could be useful?
eval_points = np.transpose(np.array(f['data']))
solver.eval_at(eval_points) # solver returns values evaluated at these points
solver.set_domain("SQUARE")

# format: [agg_stress,agg_pressure,agg_rot]
# agg_stress (e.g.) = (count,mean,M2), see welfords.py

agg_sd = [(0,0,0),(0,0,0),(0,0,0)]
agg_fd = [(0,0,0),(0,0,0),(0,0,0)]

# kap_test = 0

sol_0 = solver.run_sd(0,0)

# STOCHASTIC DOMAIN STATISTICS
start = time.time()
for i in range(0,n):
    print(i)
    kappa = generate_kappa(1)
    # kappa = -1
    print(kappa)
    sol_list = solver.run_sd(kappa,eps)
    agg_sd = update_agg(agg_sd,sol_list)
    # print(sol_list[1])
time_sd = time.time() - start

# print(agg_sd[1][1]) # print average pressure
print(time_sd)


# note: speed up in load_fd() pre-processing; map eval_points to mesh triangles.
# FIXED DOMAIN STATISTICS
start = time.time()
# solver.load_fd()
# for i in range(0,n): # NOTe UPDATE RUN_FD TO INCORPORATE SPEED UP i.e., ONLY ASSEMBLE a ONCE ETC
#     print(i)
#     kappa = generate_kappa(1)
#     print(kappa)
#     sol_list = solver.run_fd(kappa)
#     agg_fd = update_agg(agg_fd,sol_list)
time_fd = time.time() - start


# N = 5000
# idx=random.sample(range(np.size(eval_points,1)),N) # random sample of evaluation points; take it easy on the graphics

# U = agg_sd[1][1]

# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(projection='3d')
# ax.scatter(eval_points[0,idx],eval_points[1,idx],U[idx],marker=".")
# plt.show()


# # solver.plot_sol(3)
# print(time_fd)
# #print(agg_fd[1][1]) # print average pressure
# # print(agg_sd)
# # stats_sd = finalize(agg_sd) 
stats_sd = 0
# # print(agg_sd)
# # stats_fd = finalize(agg_fd)
stats_fd = 0


# save variables
# https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
save_file = '../data/eps_' + str(eps) + '.pkl'
with open(save_file,'wb') as f:
    pickle.dump([eps,n,eval_points,sol_0,agg_sd,stats_sd,agg_fd,stats_fd,time_sd,time_fd],f)
    f.close()

