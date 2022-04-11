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

def generate_kappa(a):
    # generate random epsilon: uniform distribution between [-a,a]
    return (random.random()*2-1)*a

def update_agg(agg_list,sol_list):
    for i in range(len(agg_list)):
        agg_list[i] = welfords.update(agg_list[i],sol_list[i])
    return agg_list

solver = SC_Solver(80)

f = h5py.File('eval_points.h5','r') # (!) need to set working directory; os.chdir() could be useful?
eval_points = np.transpose(np.array(f['data']))
solver.eval_at(eval_points) # solver returns values evaluated at these points
solver.set_domain("SQUARE")

eps = 0.3
n = 1

agg_sd = [(0,0,0),(0,0,0),(0,0,0)]

for i in range(0,n):
    # kappa = generate_kappa(1)
    kappa = -1
    print(kappa)
    sol_list = solver.run_sd(kappa,eps)
    agg_sd = update_agg(agg_sd,sol_list)

N = 5000
idx=random.sample(range(np.size(eval_points,1)),N) # random sample of evaluation points; take it easy on the graphics

# U = sol_list[1][:,0]
# print(agg_sd[1][1])
# print(np.transpose(sol_0[1][:,0]))
U = agg_sd[1][1]
# print(n)
# U = agg_sd[1][1]

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection='3d')
ax.scatter(eval_points[0,idx],eval_points[1,idx],U[idx],marker=".")
plt.show()