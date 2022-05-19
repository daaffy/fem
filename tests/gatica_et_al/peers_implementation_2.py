from fenics import *
from multiphenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# exec(open("peers_implementation_2.py").read())

# solid + fluid (uncoupled)

# -------------------------------------------------------------------
# f = as_vector((Expression('2*pi*pi*cos(pi*(x[0]+x[1]))', degree=2), Expression('2*pi*pi*cos(pi*(x[0]+x[1]))', degree=2)))
f = Constant((0.0,1.0))

# parameters
rho_s = 1
rho_f = 1
omega = sqrt(2)*Constant(pi)
lamb = 1
nu = 1
c = omega/(sqrt(2)*Constant(pi))
C = np.array([[2*nu+lamb, 0, 0, lamb], [0, 2*nu, 0, 0], [0, 0, 2*nu, 0], [lamb, 0, 0, 2*nu+lamb]])
invC = as_matrix(np.linalg.inv(C))
# invC_11 = as_matrix(invC[0:2,0:2]) # for d2 bilinear form
# invC_12 = as_matrix(invC[0:2,2:4])
# invC_21 = as_matrix(invC[2:4,0:2])
# invC_22 = as_matrix(invC[2:4,2:4])
kappa_s = sqrt(omega**2*rho_s)
kappa_f = sqrt(omega**2/(c**2))

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
# define function spaces 

_B = FiniteElement("Bubble", mesh.ufl_cell(), 3)
_H = FiniteElement("RT", mesh.ufl_cell(), 1)
R = FiniteElement("CG", mesh.ufl_cell(), 1)
V_1 = FunctionSpace(mesh, MixedElement([_H,_B,_H,_B,R])) # solid domain
V_2 = FunctionSpace(mesh, FiniteElement("CG", mesh.ufl_cell(), 1)) # fluid domain
# V_1 = BlockElement(_H,_B,_H,_B,R)
# V_2 = BlockElement(V_1)
V = BlockFunctionSpace([V_1,V_2], restrict=[solid_restriction,fluid_restriction])

# -------------------------------------------------------------------
# define variational problem

# trial functions
trial_func = BlockTrialFunction(V)
(pre_sig, p) = block_split(trial_func)
(sig1_H, pre_sig1_B, sig2_H, pre_sig2_B, r) = dolfin.split(pre_sig)
sig1_B = as_vector([pre_sig1_B.dx(1),-pre_sig1_B.dx(0)])
sig2_B = as_vector([pre_sig2_B.dx(1),-pre_sig2_B.dx(0)])
sigma = as_tensor((sig1_H+sig1_B,sig2_H+sig2_B))
trial = (sigma,p,r)

# # # test functions
test_func = BlockTestFunction(V)
(pre_tau, q) = block_split(test_func)
(tau1_H, pre_tau1_B, tau2_H, pre_tau2_B, s) = dolfin.split(pre_tau)
tau1_B = as_vector([pre_tau1_B.dx(1),-pre_tau1_B.dx(0)])
tau2_B = as_vector([pre_tau2_B.dx(1),-pre_tau2_B.dx(0)])
tau = as_tensor((tau1_H+tau1_B,tau2_H+tau2_B))
test = (tau,q,s)

n = FacetNormal(mesh) # (!) check this direction
g = Constant((0.0,0.0)) # homogeneous transmission condition

dx = Measure("dx")(subdomain_data=subdomains)
ds = Measure("ds")(subdomain_data=boundaries)
dS = Measure("dS")(subdomain_data=boundaries)
dS = dS(2) # restrict to the interface, which has facet ID equal to 2

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
# sgn_1 = '+' # (!!!) need to check sgn's

# def _t_s(trial,test):
#     sigma = trial[0]
#     psi = test[3]
#     return dot(sigma(sgn_1)*n(sgn_1),psi(sgn_1))*dS

# def _t_f(trial,test):
#     p = trial[1]
#     psi = test[3]
#     return dot(p(sgn_1)*n(sgn_1),psi(sgn_1))*dS

# ----- load form

def elastic_load(test):
    #(tau1_H, tau1_B, tau2_H, tau2_B, s) = tau_tens
    tau = test[0]
    return 1/kappa_s**2*dot(f,div(tau))*dx(solid_dom)

a = [[A_solid(trial,test),0],[0,A_fluid(trial,test)]]

l =  [elastic_load(test),0]

# a = [s11(sig1_H,tau1_H,invC_11)]
# # a = [[s11(sig1_H,tau1_H,invC_11), s11(sig1_B,tau1_H,invC_11), s12(sig2_H,tau1_H,invC_12), s12(sig2_B,tau1_H,invC_12), -r1(tau1_H,r,1)],
# #      [s11(sig1_H,tau1_B,invC_11), s11(sig1_B,tau1_B,invC_11), s12(sig2_H,tau1_B,invC_12), s12(sig2_B,tau1_B,invC_12), -r1(tau1_B,r,1)],
# #      [s12(sig1_H,tau2_H,invC_21), s12(sig1_B,tau2_H,invC_21), s11(sig2_H,tau2_H,invC_22), s11(sig2_B,tau2_H,invC_22), r1(tau2_H,r,0)],
# #      [s12(sig1_H,tau2_B,invC_21), s12(sig1_B,tau2_B,invC_21), s11(sig2_H,tau2_B,invC_22), s11(sig2_B,tau2_B,invC_22), r1(tau1_B,r,0)],
# #      [-omega**2*r1(sig1_H,s,1), -omega**2*r1(sig1_B,s,1), omega**2*r1(sig2_H,s,0), omega**2*r1(sig2_B,s,0), 0]]

A = block_assemble(a)
L = block_assemble(l)

# -------------------------------------------------------------------
# solve
U = BlockFunction(V)
block_solve(A, U.block_vector(), L)

# -------------------------------------------------------------------
# plot solution
(pre_sig, p) = block_split(U)
(sig1_H, pre_sig1_B, sig2_H, pre_sig2_B, r) = pre_sig.split()
sig1_B = as_vector([pre_sig1_B.dx(1),-pre_sig1_B.dx(0)])
sig2_B = as_vector([pre_sig2_B.dx(1),-pre_sig2_B.dx(0)])
sigma = as_tensor((sig1_H+sig1_B,sig2_H+sig2_B))
trial = (sigma,r)

plt.figure()
p = plot(sigma[0,0])
plt.show()

# -------------------------------------------------------------------
# error
stress_a = Expression((('pi*sin(pi*(x[0]+x[1]))+2*pi*cos(pi*x[0])*sin(pi*x[1])','pi*sin(pi*(x[0]+x[1]))'),
                    ('pi*sin(pi*(x[0]+x[1]))','pi*sin(pi*(x[0]+x[1]))+2*pi*sin(pi*x[0])*cos(pi*x[1])')), degree=2)

error = (stress_a[0,0]-(sig1_H[0]+sig1_B[0]))**2*dx
E = sqrt(abs(assemble(error)))
print(E)