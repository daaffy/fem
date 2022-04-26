import pickle 
# import sys
# sys.path.append('/Users/jackh/Documents/fem_fenics/katana_export/src')
import matplotlib.pyplot as plt
import numpy as np
import welfords
# import random
import plot
import os
# exec(open("/Users/jackh/Documents/fem_fenics/local_post/post_test.py").read())

def integrate(eval_points,z):
    weights = eval_points[3,:]
    return np.sum(z.T*weights,1)

# from katana_export
# file_name = '/Users/jackh/Documents/fem_fenics/katana_export/data/eps_0.01.pkl'
# with open(file_name, 'rb') as f: # 'eps_...'
#     eps,n,eval_points,agg_sd,time_sd = pickle.load(f)
#     agg_sd_0 = agg_sd

# --- LOAD DATA
# load eps=0.0
file_name = '/Users/jackh/Documents/fem_fenics/local_post/data/26-04/eps_0.0.pkl'
# file_name = '/Users/jackh/Documents/fem_fenics/katana_export/data/eps_0.01.pkl'
with open(file_name, 'rb') as f: # 'eps_...'
    eps,n,eval_points,agg_sd,time_sd = pickle.load(f)
    agg_sd_0 = agg_sd
    stats_sd_0 = welfords.finalize(agg_sd_0)


file_name = '/Users/jackh/Documents/fem_fenics/local_post/data/26-04/eps_0.02.pkl'
# file_name = '/Users/jackh/Documents/fem_fenics/katana_export/data/eps_0.01.pkl'
with open(file_name, 'rb') as f: # 'eps_...'
    eps,n,eval_points,agg_sd,time_sd = pickle.load(f)
    stats_sd = welfords.finalize(agg_sd)
plot.plot(eval_points,stats_sd[1][0]-stats_sd_0[1][0])
# plot.plot(eval_points,stats_sd[1][0])

# https://stackoverflow.com/questions/10377998/how-can-i-iterate-over-files-in-a-given-directory
directory = "/Users/jackh/Documents/fem_fenics/local_post/data/26-04/"

eps_list = np.array([])
temp = np.array([])

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".pkl"): 
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f: # 'eps_...'
            eps,n,eval_points,agg_sd,time_sd = pickle.load(f)
            stats_sd = welfords.finalize(agg_sd)
            # print(integrate(eval_points,stats_sd[1][0]))
            eps_list = np.append(eps_list,eps)
            temp = np.append(temp,integrate(eval_points,np.square(stats_sd[1][0]-stats_sd_0[1][0])))
        continue
     else:
        continue

print(eps_list)
print(temp)

# plt.figure(figsize=(7, 7))
# plt.scatter(eps_list,np.log(temp),marker=".")
# plt.show()