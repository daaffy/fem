from fenics import *
from multiphenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
parameters["ghost_mode"] = "shared_facet" # required by dS

# for debugging:
# exec(open("gatica_et_al.py").read())

# ANALYSIS OF THE COUPLING OF PRIMAL AND DUAL-MIXED
# FINITE ELEMENT METHODS FOR A TWO-DIMENSIONAL
# FLUID-SOLID INTERACTION PROBLEM
# Gatica et al.

# - PEERS in solid domain
# - Standard Lagrange element in fluid domain
# - Weakly coupled via Lagrange multipliers

# -------------------------------------------------------------------
# problem parameters
f = Constant((0.0,1.0)) 
rho_s = 1
rho_f = 1
omega = sqrt(2)*Constant(pi)
lamb = 1
nu = 1
c = omega/(sqrt(2)*Constant(pi))
C = np.array([[2*nu+lamb, 0, 0, lamb], [0, 2*nu, 0, 0], [0, 0, 2*nu, 0], [lamb, 0, 0, 2*nu+lamb]])
invC = as_matrix(np.linalg.inv(C)) # must be as_matrix to produce a UFL type later on down the line
kappa_s = omega**2*rho_s
kappa_f = omega**2/(c**2)

# domain IDs; check generate_domains.py
solid_dom = 2
fluid_dom = 1 # < solid_dom to get normals to point *into* fluid domain

# -------------------------------------------------------------------
# mesh
mesh = Mesh("mesh.xml")
subdomains = MeshFunction("size_t", mesh, "subdomains.xml")
boundaries = MeshFunction("size_t", mesh, "boundaries.xml")
# restrictions
interface_restriction = MeshRestriction(mesh, "interface_restriction.rtc.xml")
solid_restriction = MeshRestriction(mesh, "solid_restriction.rtc.xml")
fluid_restriction = MeshRestriction(mesh, "fluid_restriction.rtc.xml")

# -------------------------------------------------------------------
# define function spaces; see paper for details
_B = FiniteElement("Bubble", mesh.ufl_cell(), 3)
_H = FiniteElement("RT", mesh.ufl_cell(), 1)
R = FiniteElement("CG", mesh.ufl_cell(), 1)
V_1 = FunctionSpace(mesh, MixedElement([_H,_B,_H,_B,R])) # solid domain
V_2 = FunctionSpace(mesh, FiniteElement("CG", mesh.ufl_cell(), 1)) # fluid domain
V_3 = VectorFunctionSpace(mesh, "CG", 1) # transmission
V = BlockFunctionSpace([V_1, V_2, V_3], restrict=[solid_restriction, fluid_restriction, interface_restriction])

# -------------------------------------------------------------------
# define variational problem

def split_up(_func):
    # PEERS implemented following https://fenicsproject.discourse.group/t/peers-element-implementation/5710
    (pre_sig, p, phi) = block_split(_func)
    (sig1_H, pre_sig1_B, sig2_H, pre_sig2_B, r) = dolfin.split(pre_sig) # same as sig1_H = as_vector((pre_sig[0],pre_sig[1])), etc. # https://fenicsproject.org/qa/1123/nonlinear-solves-with-mixed-function-spaces/
    sig1_B = as_vector([pre_sig1_B.dx(1),-pre_sig1_B.dx(0)])
    sig2_B = as_vector([pre_sig2_B.dx(1),-pre_sig2_B.dx(0)])
    # sigma = as_matrix([[sig1_H],[sig2_H]]) + as_matrix([[sig1_B],[sig2_B]])
    sigma = as_tensor((sig1_H+sig1_B,sig2_H+sig2_B))
    return (sigma,p,r,phi)

# test functions
trial_func = BlockTrialFunction(V)
(pre_sig, p, phi) = block_split(trial_func)
(sig1_H, pre_sig1_B, sig2_H, pre_sig2_B, r) = dolfin.split(pre_sig)
sig1_B = as_vector([pre_sig1_B.dx(1),-pre_sig1_B.dx(0)])
sig2_B = as_vector([pre_sig2_B.dx(1),-pre_sig2_B.dx(0)])
# sigma = as_matrix([[sig1_H],[sig2_H]]) + as_matrix([[sig1_B],[sig2_B]])
sigma = as_tensor((sig1_H+sig1_B,sig2_H+sig2_B))
trial = (sigma,p,r,phi)
# trial = split_up(trial_func)

# test functions
test_func = BlockTestFunction(V)
(pre_tau, q, psi) = block_split(test_func)
(tau1_H, pre_tau1_B, tau2_H, pre_tau2_B, s) = dolfin.split(pre_tau)
tau1_B = as_vector([pre_tau1_B.dx(1),-pre_tau1_B.dx(0)])
tau2_B = as_vector([pre_tau2_B.dx(1),-pre_tau2_B.dx(0)])
# tau = as_matrix([[tau1_H],[tau2_H]]) + as_matrix([tau1_B,tau2_B])
tau = as_tensor((tau1_H+tau1_B,tau2_H+tau2_B))
test = (tau,q,s,psi)
# test = split_up(test_func)

n = FacetNormal(mesh) # (!) check this direction
g = Constant((0.0,0.0)) # homogeneous transmission condition

dx = Measure("dx")(subdomain_data=subdomains)
ds = Measure("ds")(subdomain_data=boundaries)
dS = Measure("dS")(subdomain_data=boundaries)
dS = dS(2) # restrict to the interface, which has facet ID equal to 2

# temp = as_tensor([[1,2],[3,4]])
# print(temp)

# ----- solid domain form

def _s_1(sigma,tau): # is using the global variable names problematic? i don't think so...
    return dot(div(sigma),div(tau))*dx(solid_dom) # (!) is div() operating on sigma & tau tensor correctly
    # return dot(div(sigma[0,:],tau[0,:]))
    # (sigma[0,0]+tau[0,0])*dx(solid_dom)

def _s_2(sigma,tau):
    # convert matrix to vector; invC acts on vector representation
    sigma_vec = as_vector((sigma[0,0],sigma[0,1],sigma[1,0],sigma[1,1]))
    tau_vec = as_vector((tau[0,0],tau[0,1],tau[1,0],tau[1,1]))
    return dot(invC*sigma_vec,tau_vec)*dx(solid_dom)

def _s_3(sigma,s):
    return (sigma[0,1]-sigma[1,0])*s*dx(solid_dom)

def A_solid(trial,test):
    sigma = trial[0]
    r = trial[2]
    tau = test[0]
    s = test[2]
    return _s_2(sigma,tau) - 1/kappa_s**2*_s_1(sigma,tau) + _s_3(sigma,s) + _s_3(tau,r)

# ----- fluid domain form

def _f_1(p,q):
    return dot(grad(p),grad(q))*dx(fluid_dom)

def _f_2(p,q):
    return p*q*dx(fluid_dom)

def A_fluid(trial,test):
    p = trial[1]
    q = test[1]
    return 1/(rho_f*omega**2)*_f_1(p,q) - kappa_f**2/(rho_f*omega**2)*_f_2(p,q)

# ----- transmission form


# assemble
a = A_fluid(trial,test)
# a = A_solid(trial,test)
A = assemble(a)