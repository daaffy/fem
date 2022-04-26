from fenics import *
from multiphenics import *
from mshr import *
import matplotlib.pyplot as plt

def plot_sol(U):

    # # plot solution
    (pre_sig, p, phi) = block_split(U)
    # (pre_sig, p) = block_split(U)
    (sig1_H, pre_sig1_B, sig2_H, pre_sig2_B, r) = dolfin.split(pre_sig)
    sig1_B = as_vector([pre_sig1_B.dx(1),-pre_sig1_B.dx(0)])
    sig2_B = as_vector([pre_sig2_B.dx(1),-pre_sig2_B.dx(0)])

    plt.figure()
    plot(p)
    plt.show()
