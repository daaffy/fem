from __future__ import print_function
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy, h5py
from multiphenics import *

# HOME OF THE SHAPE CALCULUS SOLVER

class SC_Solver:

    def __init__(self, mesh_size):
        self.mesh_size = mesh_size
        self.dom_set = "SQUARE"

    def set_domain(self, dom_set):
        self.dom_set = dom_set

    def  eval_at(self, eval_points):
        # load the evalutation points
        self.eval_points = eval_points

    def run_domain(self, kappa, eps):
        # stochastic domain code
        # ---------------------------------------------------------------------
        # MESH, SUBDOMAINS AND RESTRICTIONS

        # domain IDs
        self.solid_dom = 1
        self.fluid_dom = 2

        match self.dom_set:
            case "SQUARE":
                # Create mesh
                a = 1.0 + kappa*eps # should be equal to 1.0
                b = 2.0 + kappa*eps
                domain = Rectangle(Point(-b,-b),Point(b,b))
                domain_inside = Rectangle(Point(-a, -a), Point(a, a))
                domain.set_subdomain(self.solid_dom, domain)
                domain.set_subdomain(self.fluid_dom, domain_inside)
                mesh = generate_mesh(domain, self.mesh_size)
                self.mesh = mesh

                # Create boundaries
                class OnBoundary(SubDomain):
                    def inside(self, x, on_boundary):
                        return on_boundary
                class OnInterface(SubDomain):
                    def inside(self, x, on_boundary):
                        return (near(x[0], -a) or near(x[0], a)) and between(x[1], [-a, a]) or (near(x[1], -a) or near(x[1], a)) and between(x[0], [-a, a])

                self.boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
                on_boundary = OnBoundary()
                on_boundary.mark(self.boundaries, 1)
                on_interface = OnInterface()
                on_interface.mark(self.boundaries, 2)


                # Create restrictions
                nudge = 2*DOLFIN_EPS # look into getting rid of this...
                class Fluid(SubDomain):
                    def inside(self, x, on_boundary):
                        return between(x[0],[-a,a]) and between(x[1],[-a,a])
                class Solid(SubDomain):
                    def inside(self, x, on_boundary):
                        return not(between(x[0],[-a+nudge,a-nudge]) and between(x[1],[-a+nudge,a-nudge]))

                # Create subdomains
                self.subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains()) # 4th argument passes the domain tags to the mesh function
                solid = Solid()
                fluid = Fluid()

                self.boundary_restriction = MeshRestriction(mesh, on_boundary)
                self.interface_restriction = MeshRestriction(mesh, on_interface)
                self.fluid_restriction = MeshRestriction(mesh, fluid)
                self.solid_restriction = MeshRestriction(mesh, solid)
            
            case "CIRCLE":
                # Create mesh
                a = 1.0 + kappa*eps # should be equal to 1.0
                b = 2.0 + kappa*eps
                domain = Circle(Point(0,0), 2)
                domain_inside = Circle(Point(0,0), 1)
                domain.set_subdomain(self.solid_dom, domain)
                domain.set_subdomain(self.fluid_dom, domain_inside)
                mesh = generate_mesh(domain, self.mesh_size)
                self.mesh = mesh

                # Create boundaries
                class OnBoundary(SubDomain):
                    def inside(self, x, on_boundary):
                        return on_boundary
                class OnInterface(SubDomain):
                    def inside(self, x, on_boundary):
                        return (near(x[0], -a) or near(x[0], a)) and between(x[1], [-a, a]) or (near(x[1], -a) or near(x[1], a)) and between(x[0], [-a, a])

                self.boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
                on_boundary = OnBoundary()
                on_boundary.mark(self.boundaries, 1)
                on_interface = OnInterface()
                on_interface.mark(self.boundaries, 2)


                # Create restrictions
                nudge = 2*DOLFIN_EPS # look into getting rid of this...
                class Fluid(SubDomain):
                    def inside(self, x, on_boundary):
                        return between(x[0],[-a,a]) and between(x[1],[-a,a])
                class Solid(SubDomain):
                    def inside(self, x, on_boundary):
                        return not(between(x[0],[-a+nudge,a-nudge]) and between(x[1],[-a+nudge,a-nudge]))

                # Create subdomains
                self.subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains()) # 4th argument passes the domain tags to the mesh function
                solid = Solid()
                fluid = Fluid()

                self.boundary_restriction = MeshRestriction(mesh, on_boundary)
                self.interface_restriction = MeshRestriction(mesh, on_interface)
                self.fluid_restriction = MeshRestriction(mesh, fluid)
                self.solid_restriction = MeshRestriction(mesh, solid)

    def run_sd(self, kappa, eps):

        parameters["ghost_mode"] = "shared_facet" # required by dS

        # ---------------------------------------------------------------------
        # RUN MESH AND RESTRICTION CODE
        self.run_domain(kappa,eps)
        mesh = self.mesh

        # ---------------------------------------------------------------------
        # DEFINE FUNCTION SPACES

        _B = FiniteElement("Bubble", mesh.ufl_cell(), 3)
        _H = FiniteElement("RT", mesh.ufl_cell(), 1)
        R = FiniteElement("CG", mesh.ufl_cell(), 1)
        V_1 = FunctionSpace(mesh, MixedElement([_H,_B,_H,_B,R])) # solid domain
        V_2 = FunctionSpace(mesh, FiniteElement("CG", mesh.ufl_cell(), 1)) # fluid domain
        V_3 = VectorFunctionSpace(mesh, "CG", 1) # fluid domain
        V = BlockFunctionSpace([V_1, V_2, V_3], restrict=[self.solid_restriction, self.fluid_restriction, self.interface_restriction])

        # ---------------------------------------------------------------------
        # BUILD LINEAR FORMS

        # parameters
        rho_s = 1
        rho_f = 1
        omega = sqrt(2)*Constant(pi)
        lamb = 1
        nu = 1
        c = omega/(sqrt(2)*Constant(pi))
        C = numpy.array([[2*nu+lamb, 0, 0, lamb], [0, 2*nu, 0, 0], [0, 0, 2*nu, 0], [lamb, 0, 0, 2*nu+lamb]])
        invC = numpy.linalg.inv(C)
        invC_11 = as_matrix(invC[0:2,0:2]) # for d2 bilinear form
        invC_12 = as_matrix(invC[0:2,2:4])
        invC_21 = as_matrix(invC[2:4,0:2])
        invC_22 = as_matrix(invC[2:4,2:4])

        # -------------------------------------------------------------------
        # define variational problem

        # trial functions
        trial_func = BlockTrialFunction(V)
        (pre_sig, p, phi) = block_split(trial_func)
        # (pre_sig, p) = block_split(trial_func)
        (sig1_H, pre_sig1_B, sig2_H, pre_sig2_B, r) = dolfin.split(pre_sig)
        sig1_B = as_vector([pre_sig1_B.dx(1),-pre_sig1_B.dx(0)])
        sig2_B = as_vector([pre_sig2_B.dx(1),-pre_sig2_B.dx(0)])
        # p = trial_func[1]
        # phi = trial_func[2]

        # test functions
        test_func = BlockTestFunction(V)
        (pre_tau, q, psi) = block_split(test_func)
        # (pre_tau, q) = block_split(test_func)
        (tau1_H, pre_tau1_B, tau2_H, pre_tau2_B, s) = dolfin.split(pre_tau)
        tau1_B = as_vector([pre_tau1_B.dx(1),-pre_tau1_B.dx(0)])
        tau2_B = as_vector([pre_tau2_B.dx(1),-pre_tau2_B.dx(0)])
        # q = trial_func[1]
        # psi = trial_func[2]

        # as a compact input for elastic_form
        sig_tens = (sig1_H, sig1_B, sig2_H, sig2_B, r)
        tau_tens = (tau1_H, tau1_B, tau2_H, tau2_B, s)

        stress_a = Expression((('pi*sin(pi*(x[0]+x[1]))+2*pi*cos(pi*x[0])*sin(pi*x[1])','pi*sin(pi*(x[0]+x[1]))'),
                            ('pi*sin(pi*(x[0]+x[1]))','pi*sin(pi*(x[0]+x[1]))+2*pi*sin(pi*x[0])*cos(pi*x[1])')), degree=2)
        pressure_a = Expression('cos(pi*x[0])*cos(pi*x[1])', degree=2)

        f = [Expression('2*pi*pi*cos(pi*(x[0]+x[1]))', degree=2), 
    Expression('2*pi*pi*cos(pi*(x[0]+x[1]))', degree=2)]

        n = FacetNormal(mesh)
        g = stress_a*n + pressure_a*n
        # g = Constant((0.0,0.0))

        f_sgn = '-'
        s_sgn = '+'

        dx = Measure("dx")(subdomain_data=self.subdomains)
        ds = Measure("ds")(subdomain_data=self.boundaries)
        dS = Measure("dS")(subdomain_data=self.boundaries)
        dS = dS(2) # restrict to the interface, which has facet ID equal to 2

        solid_dom = self.solid_dom
        fluid_dom = self.fluid_dom



        # sig_form ----------------------------------------------------------------------------------------------
        def s11(sig_vec,tau_vec,invC_):
            return div(sig_vec)*div(tau_vec)/rho_s*dx(solid_dom) -omega**2*dot(invC_*sig_vec,tau_vec)*dx(solid_dom)

        def s12(sig_vec,tau_vec,invC_):
            return -omega**2*dot(invC_*sig_vec,tau_vec)*dx(solid_dom)

        def r1(sig_vec,s,i):
            return s*sig_vec[i]*dx(solid_dom)

        def elastic_form(sig_tens,tau_tens):
            (sig1_H, sig1_B, sig2_H, sig2_B, r) = sig_tens
            (tau1_H, tau1_B, tau2_H, tau2_B, s) = tau_tens
            return s11(sig1_H,tau1_H,invC_11) + s11(sig1_B,tau1_H,invC_11) + s12(sig2_H,tau1_H,invC_12) + s12(sig2_B,tau1_H,invC_12) + -r1(tau1_H,r,1) + \
            s11(sig1_H,tau1_B,invC_11) + s11(sig1_B,tau1_B,invC_11) + s12(sig2_H,tau1_B,invC_12) + s12(sig2_B,tau1_B,invC_12) + -r1(tau1_B,r,1) + \
            s12(sig1_H,tau2_H,invC_21) + s12(sig1_B,tau2_H,invC_21) + s11(sig2_H,tau2_H,invC_22) + s11(sig2_B,tau2_H,invC_22) + r1(tau2_H,r,0) + \
            s12(sig1_H,tau2_B,invC_21) + s12(sig1_B,tau2_B,invC_21) + s11(sig2_H,tau2_B,invC_22) + s11(sig2_B,tau2_B,invC_22) + r1(tau1_B,r,0) + \
            -omega**2*r1(sig1_H,s,1) + -omega**2*r1(sig1_B,s,1) + omega**2*r1(sig2_H,s,0) + omega**2*r1(sig2_B,s,0) + 0

        def elastic_load(tau_tens):
            (tau1_H, tau1_B, tau2_H, tau2_B, s) = tau_tens
            return f[0]*div(tau1_H)*dx(solid_dom) + f[0]*div(tau1_B)*dx(solid_dom) + f[1]*div(tau2_H)*dx(solid_dom) + f[1]*div(tau2_B)*dx(solid_dom) + 0

        # other forms ----------------------------------------------------------------------------------------------
        def w1(tau_vec,phi):
            return -omega**2*dot(as_vector([dot(tau_vec(s_sgn),n(s_sgn)),0]),phi(s_sgn))*dS

        def w2(tau_vec,phi):
            return -omega**2*dot(as_vector([0,dot(tau_vec(s_sgn),n(s_sgn))]),phi(s_sgn))*dS

        def b_form(sig_tens,psi):
            (sig1_H, sig1_B, sig2_H, sig2_B, r) = sig_tens
            return w1(sig1_H,psi) + w1(sig1_B,psi) + w2(sig2_H,psi) + w2(sig2_B,psi)

        def fluid_form(p,q):
            return (dot(grad(p), grad(q)) - (omega**2/(c**2*rho_f))*p*q)*dx(fluid_dom)

        def w3(p,phi):
            return -omega**2*dot(p(f_sgn)*n(s_sgn),phi(f_sgn))*dS

        a = [[elastic_form(sig_tens,tau_tens),      0,                      b_form(tau_tens,phi)],
            [0,                                    fluid_form(p,q),        w3(q,phi)],
            [b_form(sig_tens,psi),                 w3(p,psi),              0]]

        l =  [elastic_load(tau_tens), 0, -omega**2*dot(g('+'),psi('+'))*dS]

        # a = [[elastic_form(sig_tens,tau_tens),      0],
        #      [0,                                    fluid_form(p,q)]]

        # l =  [elastic_load(tau_tens), 0]

        A = block_assemble(a)
        L = block_assemble(l)

        # ---------------------------------------------------------------------
        
        # solve
        self.U = BlockFunction(V)
        block_solve(A, self.U.block_vector(), L, linear_solver="mumps")


        # -------------------------------------------------------------------
        (pre_sig, p, phi) = block_split(self.U)
        (sig1_H, pre_sig1_B, sig2_H, pre_sig2_B, r) = dolfin.split(pre_sig)
        sig1_B = as_vector([pre_sig1_B.dx(1),-pre_sig1_B.dx(0)])
        sig2_B = as_vector([pre_sig2_B.dx(1),-pre_sig2_B.dx(0)])

        eval_points = self.eval_points

        # evaluate at eval_points and export
        stress = numpy.zeros([numpy.size(eval_points,1), 4])
        pressure = numpy.zeros([numpy.size(eval_points,1), 1])
        rot = numpy.zeros([numpy.size(eval_points,1), 1])

        for i in range(numpy.size(eval_points,1)):
            x = Point(eval_points[0,i],eval_points[1,i])
            try: 
                stress[i,:] = numpy.append(sig1_H(x)+sig1_B(x), sig2_H(x)+sig2_B(x)) # throws a runtime error if the point is outside the mesh domain
                pressure[i] = p(x)
                rot[i] = r(x)
            except:
                continue

        return stress, pressure, rot

    # def run_fd(self, kappa):
    #     # ---------------------------------------------------------------------
    #     # RUN MESH AND RESTRICTION CODE
    #     self.run_domain(0,0)
    #     mesh = self.mesh

    #     # ---------------------------------------------------------------------

    #     self.run_sd(0,0)
    #     U_0 = self.U # save unperterbed solution for later

    #     # ---------------------------------------------------------------------
    #     # DEFINE FUNCTION SPACES

    #     P1 = FunctionSpace(mesh, "RT", 2) # (!) Use FiniteElement??? See UFL doc sheet
    #     P2 = FunctionSpace(mesh, "CG", 2)
    #     # P3 = VectorFunctionSpace(mesh, "CG", 2)
    #     V = BlockFunctionSpace([P1, P1, P2, P2], restrict=[self.solid_restriction, self.solid_restriction, self.solid_restriction, self.fluid_restriction])

    #     # ---------------------------------------------------------------------
    #     # BUILD LINEAR FORMS

    #     # parameters
    #     rho_s = 1
    #     rho_f = 1
    #     omega = sqrt(2)*Constant(pi)
    #     lamb = 1
    #     nu = 1
    #     c = omega/(sqrt(2)*Constant(pi))
    #     C = numpy.array([[2*nu+lamb, 0, 0, lamb], [0, 2*nu, 0, 0], [0, 0, 2*nu, 0], [lamb, 0, 0, 2*nu+lamb]])
    #     invC = numpy.linalg.inv(C)
    #     invC_11 = as_matrix(invC[0:2,0:2]) # for d2 bilinear form
    #     invC_12 = as_matrix(invC[0:2,2:4])
    #     invC_21 = as_matrix(invC[2:4,0:2])
    #     invC_22 = as_matrix(invC[2:4,2:4])

    #     # Define variational problem
    #     trial_func = BlockTrialFunction(V) # trial = "solution", test = "function to test against"
    #     (sig1, sig2, r, p) = block_split(trial_func)

    #     test_func = BlockTestFunction(V)
    #     (tau1, tau2, s, q) = block_split(test_func)

    #     stress_a = Expression((('pi*sin(pi*(x[0]+x[1]))+2*pi*cos(pi*x[0])*sin(pi*x[1])','pi*sin(pi*(x[0]+x[1]))'),
    #                         ('pi*sin(pi*(x[0]+x[1]))','pi*sin(pi*(x[0]+x[1]))+2*pi*sin(pi*x[0])*cos(pi*x[1])')), degree=2)
    #     pressure_a = Expression('cos(pi*x[0])*cos(pi*x[1])', degree=2)
    #     # f = Constant((0.0,1.0))
    #     f = Expression('2*pi*pi*cos(pi*(x[0]+x[1]))', degree=2)
    #     n = FacetNormal(mesh)
    #     g = stress_a*n + pressure_a*n

    #     temp_sgn_1 = '+' #sig etc.
    #     temp_sgn_n = '+'
    #     temp_sgn_pr = '+'
    #     temp_sgn_2 = '+' #psi/phi

    #     # MEASURES #
    #     dx = Measure("dx")(subdomain_data=self.subdomains)
    #     ds = Measure("ds")(subdomain_data=self.boundaries)
    #     dS = Measure("dS")(subdomain_data=self.boundaries)
    #     dS = dS(2) # restrict to the interface, which has facet ID equal to 2

    #     a = [[div(sig1)*div(tau1)/rho_s*dx(self.solid_dom) - omega**2*dot(invC_11*sig1,tau1)*dx(self.solid_dom), - omega**2*dot(invC_12*sig2,tau1)*dx(self.solid_dom),   -r*tau1[1]*dx(self.solid_dom) , 0],
    #         [- omega**2*dot(invC_21*sig1,tau2)*dx(self.solid_dom), div(sig2)*div(tau2)/rho_s*dx(self.solid_dom) - omega**2*dot(invC_22*sig2,tau2)*dx(self.solid_dom), r*tau2[0]*dx(self.solid_dom) , 0],
    #         [-omega**2*s*sig1[1]*dx(self.solid_dom)             , omega**2*s*sig2[0]*dx(self.solid_dom)       , 0            , 0],
    #         [0, 0, 0, (dot(grad(p), grad(q)) - (omega**2/(c**2*rho_f))*p*q)*dx(self.fluid_dom)]]

    #     l =  [f*div(tau1)*dx(self.solid_dom)                       , f*div(tau2)*dx(self.solid_dom)                       , 0 , 0]

    #     A = block_assemble(a)
    #     L = block_assemble(l)

    #     # ---------------------------------------------------------------------
    #     # Compute solution
    #     self.U = BlockFunction(V) # initialize solution function object
    #     block_solve(A, self.U.block_vector(), L) # solve linear variational problem

    #     eval_points = self.eval_points

        

    #     # evaluate at eval_points and export
    #     stress = numpy.zeros([numpy.size(eval_points,1), 4])
    #     pressure = numpy.zeros([numpy.size(eval_points,1), 1])
    #     rot = numpy.zeros([numpy.size(eval_points,1), 1])

    #     for i in range(numpy.size(eval_points,1)):
    #         x = Point(eval_points[0,i],eval_points[1,i])
    #         try: 
    #             stress[i,:] = numpy.append(self.U[0](x), self.U[1](x)) # throws a runtime error if the point is outside the mesh domain
    #             pressure[i] = self.U[3](x)
    #             rot[i] = self.U[2](x)
    #         except:
    #             continue

    #     return stress, pressure, rot
    
    # def calc_a_fd():
    def load_fd(self):
        self.run_domain(0,0)
        self.run_sd(0,0)
        self.U_0 = self.U # save unperterbed solution for later

        mesh = self.mesh
        solid_restriction = self.solid_restriction
        fluid_restriction = self.fluid_restriction
        solid_dom = self.solid_dom
        fluid_dom = self.fluid_dom
        subdomains = self.subdomains
        boundaries = self.boundaries
        eval_points = self.eval_points

        # ---------------------------------------------------------------------
        # DEFINE FUNCTION SPACES

        P1 = FunctionSpace(mesh, "RT", 2) # (!) Use FiniteElement??? See UFL doc sheet
        P2 = FunctionSpace(mesh, "CG", 2)
        P3 = VectorFunctionSpace(mesh, "CG", 2)
        V = BlockFunctionSpace([P1, P1, P2, P2], restrict=[solid_restriction, solid_restriction, solid_restriction, fluid_restriction])

        # ---------------------------------------------------------------------
        # BUILD LINEAR FORMS

        # parameters
        rho_s = 1
        rho_f = 1
        omega = sqrt(2)*Constant(pi)
        lamb = 1
        nu = 1
        c = omega/(sqrt(2)*Constant(pi))
        C = numpy.array([[2*nu+lamb, 0, 0, lamb], [0, 2*nu, 0, 0], [0, 0, 2*nu, 0], [lamb, 0, 0, 2*nu+lamb]])
        invC = numpy.linalg.inv(C)
        invC_11 = as_matrix(invC[0:2,0:2]) # for d2 bilinear form
        invC_12 = as_matrix(invC[0:2,2:4])
        invC_21 = as_matrix(invC[2:4,0:2])
        invC_22 = as_matrix(invC[2:4,2:4])

        # Define variational problem
        trial_func = BlockTrialFunction(V) # trial = "solution", test = "function to test against"
        (sig1, sig2, r, p) = block_split(trial_func)

        test_func = BlockTestFunction(V)
        (tau1, tau2, s, q) = block_split(test_func)

        # stress_a = Expression((('pi*sin(pi*(x[0]+x[1]))+2*pi*cos(pi*x[0])*sin(pi*x[1])','pi*sin(pi*(x[0]+x[1]))'),
        #                     ('pi*sin(pi*(x[0]+x[1]))','pi*sin(pi*(x[0]+x[1]))+2*pi*sin(pi*x[0])*cos(pi*x[1])')), degree=2)
        # pressure_a = Expression('cos(pi*x[0])*cos(pi*x[1])', degree=2)
        # # f = Constant((0.0,1.0))
        # f = Expression('2*pi*pi*cos(pi*(x[0]+x[1]))', degree=2)
        # n = FacetNormal(mesh)
        # g = stress_a*n + pressure_a*n

        temp_sgn_1 = '+' #sig etc.
        temp_sgn_n = '+'
        temp_sgn_pr = '+'
        temp_sgn_2 = '+' #psi/phi

        # MEASURES #
        dx = Measure("dx")(subdomain_data=subdomains)
        ds = Measure("ds")(subdomain_data=boundaries)
        dS = Measure("dS")(subdomain_data=boundaries)
        dS = dS(2) # restrict to the interface, which has facet ID equal to 2

        self.a = [[div(sig1)*div(tau1)/rho_s*dx(solid_dom) - omega**2*dot(invC_11*sig1,tau1)*dx(solid_dom), - omega**2*dot(invC_12*sig2,tau1)*dx(solid_dom),   -omega**2*r*tau1[1]*dx(solid_dom) , 0],
            [- omega**2*dot(invC_21*sig1,tau2)*dx(solid_dom), div(sig2)*div(tau2)/rho_s*dx(solid_dom) - omega**2*dot(invC_22*sig2,tau2)*dx(solid_dom), omega*2*r*tau2[0]*dx(solid_dom) , 0],
            [-s*sig1[1]*dx(solid_dom)             , s*sig2[0]*dx(solid_dom)       , 0            , 0],
            [0, 0, 0, (dot(grad(p), grad(q)) - (omega**2/(c**2*rho_f))*p*q)*dx(fluid_dom)]]

        self.A = block_assemble(self.a)

    def run_fd(self, kappa):
        # ---------------------------------------------------------------------
        # RUN MESH AND RESTRICTION CODE
        
        mesh = self.mesh
        solid_restriction = self.solid_restriction
        fluid_restriction = self.fluid_restriction
        solid_dom = self.solid_dom
        fluid_dom = self.fluid_dom
        subdomains = self.subdomains
        boundaries = self.boundaries
        eval_points = self.eval_points

        # ---------------------------------------------------------------------
        # DEFINE FUNCTION SPACES

        P1 = FunctionSpace(mesh, "RT", 2) # (!) Use FiniteElement??? See UFL doc sheet
        P2 = FunctionSpace(mesh, "CG", 2)
        P3 = VectorFunctionSpace(mesh, "CG", 2)
        V = BlockFunctionSpace([P1, P1, P2, P2], restrict=[solid_restriction, solid_restriction, solid_restriction, fluid_restriction])

        # ---------------------------------------------------------------------
        # BUILD LINEAR FORMS

        # parameters
        rho_s = 1
        rho_f = 1
        omega = sqrt(2)*Constant(pi)
        lamb = 1
        nu = 1
        c = omega/(sqrt(2)*Constant(pi))
        C = numpy.array([[2*nu+lamb, 0, 0, lamb], [0, 2*nu, 0, 0], [0, 0, 2*nu, 0], [lamb, 0, 0, 2*nu+lamb]])
        invC = numpy.linalg.inv(C)
        invC_11 = as_matrix(invC[0:2,0:2]) # for d2 bilinear form
        invC_12 = as_matrix(invC[0:2,2:4])
        invC_21 = as_matrix(invC[2:4,0:2])
        invC_22 = as_matrix(invC[2:4,2:4])

        # Define variational problem
        # trial_func = BlockTrialFunction(V) # trial = "solution", test = "function to test against"
        # (sig1, sig2, r, p) = block_split(trial_func)

        test_func = BlockTestFunction(V)
        (tau1, tau2, s, q) = block_split(test_func)

        # stress_a = Expression((('pi*sin(pi*(x[0]+x[1]))+2*pi*cos(pi*x[0])*sin(pi*x[1])','pi*sin(pi*(x[0]+x[1]))'),
        #                     ('pi*sin(pi*(x[0]+x[1]))','pi*sin(pi*(x[0]+x[1]))+2*pi*sin(pi*x[0])*cos(pi*x[1])')), degree=2)
        # pressure_a = Expression('cos(pi*x[0])*cos(pi*x[1])', degree=2)
        # # f = Constant((0.0,1.0))
        # # f = Expression('2*pi*pi*cos(pi*(x[0]+x[1]))', degree=2)
        # # tf = Expression('2*pi*pi*cos(pi*(3*x[0]+x[1]))', degree=2)
        # n = FacetNormal(mesh)
        # g = stress_a*n + pressure_a*n

        temp_sgn_1 = '+' #sig etc.
        temp_sgn_n = '+'
        temp_sgn_pr = '+'
        temp_sgn_2 = '+' #psi/phi

        # MEASURES #
        dx = Measure("dx")(subdomain_data=subdomains)
        ds = Measure("ds")(subdomain_data=boundaries)
        dS = Measure("dS")(subdomain_data=boundaries)
        # dS = dS(2) # restrict to the interface, which has facet ID equal to 2

        # a = [[div(sig1)*div(tau1)/rho_s*dx(solid_dom) - omega**2*dot(invC_11*sig1,tau1)*dx(solid_dom), - omega**2*dot(invC_12*sig2,tau1)*dx(solid_dom),   -r*tau1[1]*dx(solid_dom) , 0],
        #     [- omega**2*dot(invC_21*sig1,tau2)*dx(solid_dom), div(sig2)*div(tau2)/rho_s*dx(solid_dom) - omega**2*dot(invC_22*sig2,tau2)*dx(solid_dom), r*tau2[0]*dx(solid_dom) , 0],
        #     [-omega**2*s*sig1[1]*dx(solid_dom)             , omega**2*s*sig2[0]*dx(solid_dom)       , 0            , 0],
        #     [0, 0, 0, (dot(grad(p), grad(q)) - (omega**2/(c**2*rho_f))*p*q)*dx(fluid_dom)]]

        (sig1, sig2, r, p, phi) = block_split(self.U_0)

        sy1 = '-'
        # (dot(grad(p(sy1)), grad(q(sy1))) - (omega**2/(c**2*rho_f))*p(sy1)*q(sy1))*dS(2)
        l =  [0                 , 0                      , 0 , kappa*(dot(grad(p(sy1)), grad(q(sy1))) - (omega**2/(c**2*rho_f))*p(sy1)*q(sy1))*dS(2)]

        # A = block_assemble(self.a)
        L = block_assemble(l)
        
        # 
        
        
        # ---------------------------------------------------------------------
        # Compute solution
        self.U = BlockFunction(V) # initialize solution function object
        block_solve(self.A, self.U.block_vector(), L) # solve linear variational problem
        # return 0,0,0
        # ---------------------------------------------------------------------
        # PLOT SOLUTION
        # plt.figure()
        # p = plot(self.U[3])
        # plt.show()

        # return self.U

        # ---------------------------------------------------------------------
        # evaluate at eval_points and export
        stress = numpy.zeros([numpy.size(eval_points,1), 4])
        pressure = numpy.zeros([numpy.size(eval_points,1), 1])
        rot = numpy.zeros([numpy.size(eval_points,1), 1])

        for i in range(numpy.size(eval_points,1)):
            x = Point(eval_points[0,i],eval_points[1,i])
            try: 
                stress[i,:] = numpy.append(self.U[0](x), self.U[1](x)) # throws a runtime error if the point is outside the mesh domain
                pressure[i] = self.U[3](x)
                rot[i] = self.U[2](x)
            except:
                continue

        return stress, pressure, rot

    # def print_run_info(kappa):
    #     print('')

    def plot_sol(self,component):
        # PLOT SOLUTION
        plt.figure()
        p = plot(self.U[component])
        plt.show()