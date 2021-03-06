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
directory = "/Users/jackh/Documents/fem_fenics/local_post/data/20-05_01/"
file_name = directory+"eps_0.1.pkl"
# file_name = '/Users/jackh/Documents/fem_fenics/katana_export/data/eps_0.01.pkl'
with open(file_name, 'rb') as f: # 'eps_...'
    eps,n,eval_points,agg_sd,time_sd,kappa_history = pickle.load(f)
    agg_sd.finalise()
    stats_sd_0 = agg_sd
    # stats_sd_0 = welfords.finalize(agg_sd_0)

plot.plot(eval_points,stats_sd_0[0].statistics[0][:,0])

# file_name = '/Users/jackh/Documents/fem_fenics/local_post/data/16-05/eps_0.0.pkl'
# # file_name = '/Users/jackh/Documents/fem_fenics/katana_export/data/eps_0.01.pkl'
# with open(file_name, 'rb') as f: # 'eps_...'
#     eps,n,eval_points,agg_sd,time_sd = pickle.load(f)
#     stats_sd = welfords.finalize(agg_sd)
# # plot.plot(eval_points,stats_sd[1][0]-stats_sd_0[1][0])
# plot.plot(eval_points,stats_sd[1][0])

# # https://stackoverflow.com/questions/10377998/how-can-i-iterate-over-files-in-a-given-directory


eps_list = np.array([])
temp = np.array([])
avg_list = np.array([])

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".pkl"): 
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f: # 'eps_...'
            eps,n,eval_points,agg_sd,time_sd,kappa_history = pickle.load(f)
            # stats_sd = welfords.finalize(agg_sd)
            # print(integrate(eval_points,stats_sd[1][0]))
            agg_sd.finalise()
            stats_sd = agg_sd

            eps_list = np.append(eps_list,eps)
            delt = stats_sd[1].statistics[0]-stats_sd_0[1].statistics[0]
            y = np.sqrt(integrate(eval_points,np.square(delt),fluid_mask))
            # y = np.sqrt(np.square(delt[30]))
            temp = np.append(temp,y)
            # avg_list = np.append(avg_list,np.average(kappa_history))
            # print(np.average(kappa_history))
        continue
     else:
        continue

s = sorted(zip(eps_list,temp))
eps_list = [x for x,_ in s]
y = [x for _,x in s]

# y = [np.power(x,2) for x in eps_list]

# rate of convergence
r = [np.log(y[i+1]/y[i])/np.log(eps_list[i+1]/eps_list[i]) for i in range(1,len(eps_list)-1)] # rate of convergence
print(r)

# plt.plot(eps_list[1:-1],r)
# plt.plot(np.log(eps_list),np.log(y))
plt.plot(eps_list,y)
# plt.plot(eps_list,avg_list)
plt.show()

# print(eps_list)
# print(y)
# print(r)


# plt.figure(figsize=(7, 7))
# plt.scatter(eps_list,np.log(temp),marker=".")
# plt.show()