import sys
# import multiprocessing
from multiprocessing import Pool
import random
from experiment import *
import h5py
import numpy as np
from experiment import *
from sc_solver_2 import SC_Solver
# import ...

# exec(open("run.py").read())
def generate_kappa(a):
    # generate random kappa: uniform distribution between [-a,a]
    # kappa identified with omega for simplicity (see 2.2)
    return (random.random()*2-1)*a

def test(eps,n,eval_points,solver,seed=None): # initialise experiment in main and pass as an argument??
    if (seed is not None):
        random.seed(seed)

    e = experiment(eps,eval_points) # intialise experiment object

    e.begin()
    for i in range(n):
        kappa = generate_kappa(1)
        sol_list = solver.test_sol(kappa*eps)
        e.update(kappa,sol_list)

    e.end()

    return e

def run_sd(x):
    return


# print("out here")

# multiprocessing here
if __name__ == "__main__":

    # print("in here")

    # (!) does this belong outside if statement??
    arg_num = len(sys.argv)
    is_local = True
    match arg_num:
        case 3: # use given parameter values
            eps = float(sys.argv[1])
            n = int(sys.argv[2])
            process_num = 6
            is_local = False
        case _: # debug mode automatically executed when run on VSCode
            eps = 1
            n = 2
            process_num = 1
    
    # load eval_points
    f = h5py.File('eval_points.h5','r') # (!) need to set working directory; os.chdir() could be useful?
    eval_points = np.transpose(np.array(f['data']))

    solver = SC_Solver(eval_points, 100)

    with Pool(process_num) as p:
        arg_list = [(eps,n,eval_points,solver,i) for i in range(process_num)]
        e_list = p.starmap(test,arg_list)

    e = experiment.combine(e_list)

    # --- post ---
    # e1 = e_list[0]
    # e2 = e_list[1]

    # # # testing random seeding
    # print(e1.eps)
    # print(e2.eps)

