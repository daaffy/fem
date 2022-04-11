from __future__ import print_function
from fenics import *
from mshr import *
# import matplotlib.pyplot as plt
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
                nudge = 0.01 # look into getting rid of this...
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
                nudge = 0.01 # look into getting rid of this...
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

        # ---------------------------------------------------------------------
        # RUN MESH AND RESTRICTION CODE
        self.run_domain(kappa,eps)
        mesh = self.mesh

        # ---------------------------------------------------------------------
        # DEFINE FUNCTION SPACES

        P1 = FunctionSpace(mesh, "RT", 2) # (!) Use FiniteElement??? See UFL doc sheet
        P2 = FunctionSpace(mesh, "CG", 2)
        P3 = VectorFunctionSpace(mesh, "CG", 2)
        V = BlockFunctionSpace([P1, P1, P2, P2, P3], restrict=[self.solid_restriction, self.solid_restriction, self.solid_restriction, self.fluid_restriction, self.interface_restriction])


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
        (sig1, sig2, r, p, phi) = block_split(trial_func)

        test_func = BlockTestFunction(V)
        (tau1, tau2, s, q, psi) = block_split(test_func)

        stress_a = Expression((('pi*sin(pi*(x[0]+x[1]))+2*pi*cos(pi*x[0])*sin(pi*x[1])','pi*sin(pi*(x[0]+x[1]))'),
                            ('pi*sin(pi*(x[0]+x[1]))','pi*sin(pi*(x[0]+x[1]))+2*pi*sin(pi*x[0])*cos(pi*x[1])')), degree=2)
        pressure_a = Expression('cos(pi*x[0])*cos(pi*x[1])', degree=2)
        # f = Constant((0.0,1.0))

        # f = Expression('2*pi*pi*cos(pi*(x[0]+x[1]))', degree=2)

        n = FacetNormal(mesh) 

        setfg = '2'

        match setfg:
            case '1':
                f = (Expression('2*pi*pi*cos(pi*(x[0]+x[1]))', degree=2),Expression('2*pi*pi*cos(pi*(x[0]+x[1]))', degree=2))
                g = stress_a*n + pressure_a*n
            case '2':
                f = (Constant(0.0),Constant(1.0))
                g = Constant((0.0,0.0))
        
        

        temp_sgn_1 = '+' #sig etc.
        temp_sgn_n = '+'
        temp_sgn_pr = '+'
        temp_sgn_2 = '+' #psi/phi

        # MEASURES #
        dx = Measure("dx")(subdomain_data=self.subdomains)
        ds = Measure("ds")(subdomain_data=self.boundaries)
        dS = Measure("dS")(subdomain_data=self.boundaries)
        dS = dS(2) # restrict to the interface, which has facet ID equal to 2

        a = [[div(sig1)*div(tau1)/rho_s*dx(self.solid_dom) - omega**2*dot(invC_11*sig1,tau1)*dx(self.solid_dom), - omega**2*dot(invC_12*sig2,tau1)*dx(self.solid_dom),   -r*tau1[1]*dx(self.solid_dom) , 0, -omega**2*dot(as_vector([dot(tau1(temp_sgn_1),n(temp_sgn_n)),0]),phi(temp_sgn_2))*dS],
            [- omega**2*dot(invC_21*sig1,tau2)*dx(self.solid_dom), div(sig2)*div(tau2)/rho_s*dx(self.solid_dom) - omega**2*dot(invC_22*sig2,tau2)*dx(self.solid_dom), r*tau2[0]*dx(self.solid_dom) , 0, -omega**2*dot(as_vector([0,dot(tau2(temp_sgn_1),n(temp_sgn_n))]),phi(temp_sgn_2))*dS],
            [-omega**2*s*sig1[1]*dx(self.solid_dom)             , omega**2*s*sig2[0]*dx(self.solid_dom)       , 0            , 0       , 0],
            [0, 0, 0, (dot(grad(p), grad(q)) - (omega**2/(c**2*rho_f))*p*q)*dx(self.fluid_dom), -omega**2*dot(q(temp_sgn_pr)*n(temp_sgn_n),phi(temp_sgn_2))*dS],
            [-omega**2*dot(as_vector([dot(sig1(temp_sgn_1),n(temp_sgn_n)),0]),psi(temp_sgn_2))*dS, -omega**2*dot(as_vector([0,dot(sig2(temp_sgn_1),n(temp_sgn_n))]),psi(temp_sgn_2))*dS, 0, -omega**2*dot(p(temp_sgn_pr)*n(temp_sgn_n),psi(temp_sgn_2))*dS, 0]]

        l =  [f[0]*div(tau1)*dx(self.solid_dom)                       , f[1]*div(tau2)*dx(self.solid_dom)                       , 0 , 0, -omega**2*dot(g('+'),psi('+'))*dS                ]

        A = block_assemble(a)
        L = block_assemble(l)

        # ---------------------------------------------------------------------
        # Compute solution
        self.U = BlockFunction(V) # initialize solution function object
        block_solve(A, self.U.block_vector(), L) # solve linear variational problem

        # ---------------------------------------------------------------------
        
        eval_points = self.eval_points

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

    # def plot_sol(self,component):
    #     # PLOT SOLUTION
    #     plt.figure()
    #     p = plot(self.U[component])
    #     plt.show()