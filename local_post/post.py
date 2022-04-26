import pickle 
# import sys
# sys.path.append('/Users/jackh/Documents/fem_fenics/katana_export/src')
import matplotlib.pyplot as plt
import numpy as np
import welfords
import random


# --- LOAD DATA
# file_name = '/Users/jackh/Documents/fem_fenics/local_post/data/eps_0.01.pkl'
file_name = '/Users/jackh/Documents/fem_fenics/katana_export/data/eps_0.01.pkl'
with open(file_name, 'rb') as f: # 'eps_...'
    eps,n,eval_points,agg_sd,time_sd = pickle.load(f)


# --- TEST FINALIZE
def finalize(agg_list):
    for i in range(len(agg_list)):
        agg_list[i] = welfords.finalize(agg_list[i])
    return agg_list


# stats_sd = finalize(agg_sd)

# # ---- PLOTTING
# N = 10000
# idx=random.sample(range(np.size(eval_points,1)),N) # random sample of evaluation points; take it easy on the graphics

# #U = sol_0[1][:,0]

# # print(agg_sd[1][1])
# # print(np.transpose(sol_0[1][:,0]))

# # U = agg_sd[0][1][:,0]
# # U = agg_sd[1][1]

# U = stats_sd[0][0][:,1]

# # print(n)
# # U = agg_sd[1][1]

# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(projection='3d')
# ax.scatter(eval_points[0,idx],eval_points[1,idx],U[idx],marker=".")
# plt.show()