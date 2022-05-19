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

def integrate(eval_points,z,mask):
    weights = eval_points[3,:]
    for i in range(len(weights)):
        weights[i] *= mask(eval_points[0:2,i])
    return np.sum(z.T*weights,1)

def null_mask(point):
    return True

def fluid_mask(point):
    x = point[0]
    y = point[1]
    c = 0.5
    if (np.abs(x)<=c and np.abs(y) <= c):
        return True
    else:
        return False


# from katana_export
# file_name = '/Users/jackh/Documents/fem_fenics/katana_export/data/eps_0.01.pkl'
# with open(file_name, 'rb') as f: # 'eps_...'
#     eps,n,eval_points,agg_sd,time_sd = pickle.load(f)
#     agg_sd_0 = agg_sd

# --- LOAD DATA
# load eps=0.0
file_name = '/Users/jackh/Documents/fem_fenics/local_post/data/16-05/eps_0.0.pkl'
# file_name = '/Users/jackh/Documents/fem_fenics/katana_export/data/eps_0.01.pkl'
with open(file_name, 'rb') as f: # 'eps_...'
    eps,n,eval_points,agg_sd,time_sd = pickle.load(f)
    agg_sd_0 = agg_sd
    stats_sd_0 = welfords.finalize(agg_sd_0)


file_name = '/Users/jackh/Documents/fem_fenics/local_post/data/16-05/eps_0.0.pkl'
# file_name = '/Users/jackh/Documents/fem_fenics/katana_export/data/eps_0.01.pkl'
with open(file_name, 'rb') as f: # 'eps_...'
    eps,n,eval_points,agg_sd,time_sd = pickle.load(f)
    stats_sd = welfords.finalize(agg_sd)
# plot.plot(eval_points,stats_sd[1][0]-stats_sd_0[1][0])
plot.plot(eval_points,stats_sd[1][0])

# https://stackoverflow.com/questions/10377998/how-can-i-iterate-over-files-in-a-given-directory
directory = "/Users/jackh/Documents/fem_fenics/local_post/data/16-05/"

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
            y = np.sqrt(integrate(eval_points,np.square(stats_sd[1][0]-stats_sd_0[1][0]),fluid_mask))
            temp = np.append(temp,y)
        continue
     else:
        continue

s = sorted(zip(eps_list,temp))
eps_list = [x for x,_ in s]
y = [x for _,x in s]
# y = [np.power(x,2) for x in eps_list]
r = [np.log(y[i+1]/y[i])/np.log(eps_list[i+1]/eps_list[i]) for i in range(len(eps_list)-1)] # rate of convergence

print(eps_list)
print(y)
print(r)


# plt.figure(figsize=(7, 7))
# plt.scatter(eps_list,np.log(temp),marker=".")
# plt.show()