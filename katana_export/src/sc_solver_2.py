from fenics import *
from multiphenics import *
from mshr import *
# import matplotlib.pyplot as plt
import numpy as np

class SC_Solver:

    def __init__(self, eval_points, h_max):
        self.eval_points = eval_points
        self.h_max = h_max
    
    def run_sd(self, pert):
        parameters["ghost_mode"] = "shared_facet" # required by dS
        parameters['form_compiler']['no-evaluate_basis_derivatives'] = False # required? (for calculating curl later)

        # -------------------------------------------------------------------
        # parameters
        # f = [Expression('2*pi*pi*cos(pi*(x[0]+x[1]))', degree=3), 
        #     Expression('2*pi*pi*cos(pi*(x[0]+x[1]))', degree=3)]
        
        f = Constant((0.0,1.0))
        # f = Constant((1.0,0.0))

        stress_a = Expression((('pi*sin(pi*(x[0]+x[1]))+2*pi*cos(pi*x[0])*sin(pi*x[1])','pi*sin(pi*(x[0]+x[1]))'),
                            ('pi*sin(pi*(x[0]+x[1]))','pi*sin(pi*(x[0]+x[1]))+2*pi*sin(pi*x[0])*cos(pi*x[1])')), degree=2)
        pressure_a = Expression('cos(pi*x[0])*cos(pi*x[1])', degree=2)

        rho_s = 1
        rho_f = 1
        omega = sqrt(2)*Constant(pi)
        lamb = 1
        nu = 1
        c = omega/(sqrt(2)*Constant(pi))
        C = np.array([[2*nu+lamb, 0, 0, lamb], [0, 2*nu, 0, 0], [0, 0, 2*nu, 0], [lamb, 0, 0, 2*nu+lamb]])
        invC = np.linalg.inv(C)
        invC_11 = as_matrix(invC[0:2,0:2]) # for d2 bilinear form
        invC_12 = as_matrix(invC[0:2,2:4])
        invC_21 = as_matrix(invC[2:4,0:2])
        invC_22 = as_matrix(invC[2:4,2:4])

        # -------------------------------------------------------------------
        # define the domain; "square annulus"

        # domain IDs
        solid_dom = 1
        fluid_dom = 2

        eps = pert
        a = 1.0 + eps
        b = 2.0 + eps
        domain = Rectangle(Point(-b,-b), Point(b,b))
        domain_inside = Rectangle(Point(-a,-a), Point(a,a))
        # domain = domain_outside - domain_inside
        domain.set_subdomain(solid_dom, domain)
        domain.set_subdomain(fluid_dom, domain_inside)
        mesh = generate_mesh(domain, self.h_max)
        self.mesh = mesh

        class OnBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary
        class OnInterface(SubDomain):
            def inside(self, x, on_boundary):
                return (near(x[0], -a) or near(x[0], a)) and between(x[1], [-a, a]) or (near(x[1], -a) or near(x[1], a)) and between(x[0], [-a, a])


        # edge_markers = MeshFunction("bool", mesh, mesh.topology().dim() - 1)
        # OnInterface().mark(edge_markers, True)

        # mesh = refine(mesh, edge_markers)

        # plt.figure()
        # plot(mesh)
        # plt.show()

        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        on_boundary = OnBoundary()
        on_boundary.mark(boundaries, 1)
        on_interface = OnInterface()
        on_interface.mark(boundaries, 2)

        nudge = 2*DOLFIN_EPS # look into getting rid of this...
        class Fluid(SubDomain):
            def inside(self, x, on_boundary):
                return between(x[0],[-a,a]) and between(x[1],[-a,a])
        class Solid(SubDomain):
            def inside(self, x, on_boundary):
                return not(between(x[0],[-a+nudge,a-nudge]) and between(x[1],[-a+nudge,a-nudge]))

        # create subdomains
        subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains()) # 4th argument passes the domain tags to the mesh function

        # restrictions
        boundary_restriction = MeshRestriction(mesh, on_boundary)
        interface_restriction = MeshRestriction(mesh, on_interface)
        solid = Solid()
        solid_restriction = MeshRestriction(mesh, solid)
        fluid = Fluid()
        fluid_restriction = MeshRestriction(mesh, fluid)

        # file = File("subdomains.pvd")
        # file << subdomains
        # File("boundaries.pvd") << boundaries
        # File("mesh.xml") << mesh

        # -------------------------------------------------------------------
        # define function spaces 

        _B = FiniteElement("Bubble", mesh.ufl_cell(), 3)
        _H = FiniteElement("RT", mesh.ufl_cell(), 2)
        R = FiniteElement("CG", mesh.ufl_cell(), 1)
        V_1 = FunctionSpace(mesh, MixedElement([_H,_B,_H,_B,R])) # solid domain
        self.V_1 = V_1
        V_2 = FunctionSpace(mesh, FiniteElement("CG", mesh.ufl_cell(), 1)) # fluid domain
        V_3 = VectorFunctionSpace(mesh, "CG", 2) # transmission
        V = BlockFunctionSpace([V_1, V_2, V_3], restrict=[solid_restriction, fluid_restriction, interface_restriction])
        # V = BlockFunctionSpace([V_1, V_2], restrict=[solid_restriction, fluid_restriction])


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

        n = FacetNormal(mesh)
        # n = as_vector((n[0],n[1]))
        # g = stress_a*n + pressure_a*n
        g = Constant((0.0,0.0))
        # File("normal.pvd") << n

        f_sgn = '-'
        s_sgn = '+'

        dx = Measure("dx")(subdomain_data=subdomains)
        ds = Measure("ds")(subdomain_data=boundaries)
        dS = Measure("dS")(subdomain_data=boundaries)
        dS = dS(2) # restrict to the interface, which has facet ID equal to 2

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

        # -------------------------------------------------------------------
        # solve
        U = BlockFunction(V)
        block_solve(A, U.block_vector(), L, linear_solver="mumps")
        # -------------------------------------------------------------------

        # (pre_sig, p, phi) = block_split(U)
        # pressure_a = Expression('cos(pi*x[0])*cos(pi*x[1])', degree=2)
        # error = (pressure_a-p)**2*dx(2)
        # E = sqrt(abs(assemble(error)))
        # print(E)

        return U

    def eval_sol(self, U):
        eval_points = self.eval_points

        (pre_sig, p, phi) = block_split(U)
        (sig1_H, pre_sig1_B, sig2_H, pre_sig2_B, r) = pre_sig.split()
        # sig1_B = as_vector([pre_sig1_B.dx(1),-pre_sig1_B.dx(0)])
        # sig2_B = as_vector([pre_sig2_B.dx(1),-pre_sig2_B.dx(0)])

        # evaluate at eval_points and export
        stress = np.zeros([np.size(eval_points,1), 4])
        pressure = np.zeros([np.size(eval_points,1), 1])
        rot = np.zeros([np.size(eval_points,1), 1])

        # test
        # x = np.array([1.5,0.0])
        # print(self.eval_curl(pre_sig1_B,1,x))
        # print(sig1_H(Point(*x)))
        # print(self.eval_curl(pre_sig1_B,1,x)+sig1_H(Point(*x)))

        for i in range(np.size(eval_points,1)):
            x = np.array([eval_points[0,i],eval_points[1,i]])
            x_point = Point(*x)

            # (!) wrap with try/except in case point is outside the domain
            try:
                # temp = pre_sig(x) # need to evaluate from pre_sig; (!) how to incorporate the curl of the bubble elements (relatively low magnitude values)
                # (!!!!) https://bitbucket.org/fenics-project/dolfin/issues/194/split-u-versus-usplit use .split()
                stress[i,:] = np.append(sig1_H(x_point)+self.eval_curl(pre_sig1_B,1,x),sig2_H(x_point)+self.eval_curl(pre_sig2_B,2,x))
                pressure[i] = p(x_point)
                rot[i] = r(x_point)
                # rot[i] = temp[6]
            except:
                continue # need more intelligent exception handling here
        
        return stress, pressure, rot

    def eval_curl(self,f,i,x):
        if i == 1:
            el = self.V_1.sub(1).element()
        elif i == 2:
            el = self.V_1.sub(3).element()
        else:
            sys.exit("invalid index for eval_curl")

        # find the cell containing x_point
        x_point = Point(*x) 
        cell_id = self.mesh.bounding_box_tree().compute_first_entity_collision(x_point)
        cell = Cell(self.mesh, cell_id)
        coordinate_dofs = cell.get_vertex_coordinates()

        values = el.evaluate_basis_derivatives_all(1, x, coordinate_dofs, cell.orientation()).reshape((-1,2)) # reshape parameters correct?
        dofs = self.V_1.sub(1).dofmap().cell_dofs(cell_id)
        dofs = f.vector()[dofs]

        values = el.evaluate_basis_derivatives_all(1, x, coordinate_dofs, cell.orientation()).reshape((-1,2)) # reshape parameters correct?
        dofs = self.V_1.sub(1).dofmap().cell_dofs(cell_id)
        dofs = f.vector()[dofs]

        return np.array([np.sum(values[:,1]*dofs),-np.sum(values[:,0]*dofs)])

    # def plot_sol(self,U):
    #     (pre_sig, p, phi) = block_split(U)
    #     (sig1_H, pre_sig1_B, sig2_H, pre_sig2_B, r) = dolfin.split(pre_sig)
    #     sig1_B = as_vector([pre_sig1_B.dx(1),-pre_sig1_B.dx(0)])
    #     sig2_B = as_vector([pre_sig2_B.dx(1),-pre_sig2_B.dx(0)])

    #     plt.figure()
    #     plot(r) # e.g.
    #     plt.show()