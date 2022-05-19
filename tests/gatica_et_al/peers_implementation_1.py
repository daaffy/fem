from fenics import *
from multiphenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# exec(open("peers_implementation_1.py").read())

# constructing peers element on elastic annulus only

# -------------------------------------------------------------------
f = as_vector((Expression('2*pi*pi*cos(pi*(x[0]+x[1]))', degree=2), Expression('2*pi*pi*cos(pi*(x[0]+x[1]))', degree=2)))
# f = Constant((0.0,1.0))

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

# -------------------------------------------------------------------
# define the domain; "square annulus"
eps = -0.0
a = 1.0 + eps
b = 2.0 + eps
domain_outside = Rectangle(Point(-b,-b), Point(b,b))
domain_inside = Rectangle(Point(-a,-a), Point(a,a))
domain = domain_outside - domain_inside
mesh = generate_mesh(domain, 100)

# plt.figure()
# plot(mesh)
# plt.show()

# -------------------------------------------------------------------
# define function spaces 

_B = FiniteElement("Bubble", mesh.ufl_cell(), 3)
_H = FiniteElement("RT", mesh.ufl_cell(), 1)
R = FiniteElement("CG", mesh.ufl_cell(), 1)
V_1 = FunctionSpace(mesh, MixedElement([_H,_B,_H,_B,R]))

# V_1 = BlockElement(_H,_B,_H,_B,R)
# V_2 = BlockElement(V_1)
V = BlockFunctionSpace([V_1])

# -------------------------------------------------------------------
# define variational problem

# trial functions
trial_func = BlockTrialFunction(V)
# sig1_H = trial_func
# pre_sig1_B = trial_func[0][1] 
# sig2_H = trial_func[2]
# pre_sig2_B = trial_func[3]
# r = trial_func[4]
(sig1_H, pre_sig1_B, sig2_H, pre_sig2_B, r) = dolfin.split(trial_func[0])
sig1_B = as_vector([pre_sig1_B.dx(1),-pre_sig1_B.dx(0)])
sig2_B = as_vector([pre_sig2_B.dx(1),-pre_sig2_B.dx(0)])
sigma = as_tensor((sig1_H+sig1_B,sig2_H+sig2_B))
trial = (sigma,r)

# # # test functions
test_func = BlockTestFunction(V)
# tau1_H = test_func[0][0]
# pre_tau1_B = test_func[1]
# tau2_H = test_func[2]
# pre_tau2_B = test_func[3]
# s = test_func[4]
(tau1_H, pre_tau1_B, tau2_H, pre_tau2_B, s) = dolfin.split(test_func[0])
tau1_B = as_vector([pre_tau1_B.dx(1),-pre_tau1_B.dx(0)])
tau2_B = as_vector([pre_tau2_B.dx(1),-pre_tau2_B.dx(0)])
tau = as_tensor((tau1_H+tau1_B,tau2_H+tau2_B))
test = (tau,s)

# as a compact input for elastic_form
sig_tens = (sig1_H, sig1_B, sig2_H, sig2_B, r)
tau_tens = (tau1_H, tau1_B, tau2_H, tau2_B, s)

# ----- solid domain form

def _s_1(sigma,tau): # is using the global variable names problematic? i don't think so...
    return dot(div(sigma),div(tau))*dx # (!) is div() operating on sigma & tau tensor correctly
    # return dot(div(sigma[0,:],tau[0,:]))
    # (sigma[0,0]+tau[0,0])*dx(solid_dom)

def _s_2(sigma,tau):
    # convert matrix to vector; invC acts on vector representation
    sigma_vec = as_vector((sigma[0,0],sigma[0,1],sigma[1,0],sigma[1,1]))
    tau_vec = as_vector((tau[0,0],tau[0,1],tau[1,0],tau[1,1]))
    return dot(invC*sigma_vec,tau_vec)*dx

def _s_3(sigma,s):
    return (sigma[0,1]-sigma[1,0])*s*dx

def A_solid(trial,test):
    sigma = trial[0]
    r = trial[1]
    tau = test[0]
    s = test[1]
    return _s_2(sigma,tau) - 1/kappa_s**2*_s_1(sigma,tau) + _s_3(sigma,s) + _s_3(tau,r)

def elastic_load(test):
    #(tau1_H, tau1_B, tau2_H, tau2_B, s) = tau_tens
    tau = test[0]
    return 1/kappa_s**2*dot(f,div(tau))*dx


# a = [[s11(sig1_H,tau1_H,invC_11) + s11(sig1_B,tau1_H,invC_11) + s12(sig2_H,tau1_H,invC_12) + s12(sig2_B,tau1_H,invC_12) + -r1(tau1_H,r,1) + \
#     s11(sig1_H,tau1_B,invC_11) + s11(sig1_B,tau1_B,invC_11) + s12(sig2_H,tau1_B,invC_12) + s12(sig2_B,tau1_B,invC_12) + -r1(tau1_B,r,1) + \
#     s12(sig1_H,tau2_H,invC_21) + s12(sig1_B,tau2_H,invC_21) + s11(sig2_H,tau2_H,invC_22) + s11(sig2_B,tau2_H,invC_22) + r1(tau2_H,r,0) + \
#     s12(sig1_H,tau2_B,invC_21) + s12(sig1_B,tau2_B,invC_21) + s11(sig2_H,tau2_B,invC_22) + s11(sig2_B,tau2_B,invC_22) + r1(tau1_B,r,0) + \
#     -omega**2*r1(sig1_H,s,1) + -omega**2*r1(sig1_B,s,1) + omega**2*r1(sig2_H,s,0) + omega**2*r1(sig2_B,s,0) + 0]]

a = [[A_solid(trial,test)]]

l =  [elastic_load(test)]

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
(sig1_H, pre_sig1_B, sig2_H, pre_sig2_B, r) = dolfin.split(U[0])
sig1_B = as_vector([pre_sig1_B.dx(1),-pre_sig1_B.dx(0)])
sig2_B = as_vector([pre_sig2_B.dx(1),-pre_sig2_B.dx(0)])
sigma = as_tensor((sig1_H+sig1_B,sig2_H+sig2_B))
trial = (sigma,r)

plt.figure()
p = plot(sigma[0,1])
plt.show()

# -------------------------------------------------------------------
# error
stress_a = Expression((('pi*sin(pi*(x[0]+x[1]))+2*pi*cos(pi*x[0])*sin(pi*x[1])','pi*sin(pi*(x[0]+x[1]))'),
                    ('pi*sin(pi*(x[0]+x[1]))','pi*sin(pi*(x[0]+x[1]))+2*pi*sin(pi*x[0])*cos(pi*x[1])')), degree=2)

error = (stress_a[0,0]-(sig1_H[0]+sig1_B[0]))**2*dx
E = sqrt(abs(assemble(error)))
print(E)