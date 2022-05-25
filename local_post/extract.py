import numpy as np
import pickle
import sys
import os

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

def extract_all(path_str):
    directory = path_str

    eps_list = np.array([])
    temp = np.array([])

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
                delt = stats_sd[0].statistics[0]-stats_sd_0[0].statistics[0]
                y = np.sqrt(integrate(eval_points,np.square(delt),fluid_mask))
                # y = np.sqrt(np.square(delt[30]))
                temp = np.append(temp,y)
            continue
        else:
            continue

        s = sorted(zip(eps_list,temp))
        eps_list = [x for x,_ in s]
        y = [x for _,x in s]