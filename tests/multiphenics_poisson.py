from fenics import *
from multiphenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# annulus test, poisson
def test_1():
    # test_1
    # fenics only

    # -------------------------------------------------------------------
    # define the domain; "square annulus"
    eps = -0.0
    a = 1.0 + eps
    b = 2.0 + eps
    domain_outside = Rectangle(Point(-b,-b), Point(b,b))
    domain_inside = Rectangle(Point(-a,-a), Point(a,a))
    domain = domain_outside - domain_inside
    mesh = generate_mesh(domain, 100)

    class Boundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    Boundary().mark(boundaries, 1)

    # -------------------------------------------------------------------
    # define function spaces
    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx + u*v*dx
    l = v*dx

    A = assemble(a)
    L = assemble(l)
    
    bc = DirichletBC(V, Constant(0.), boundaries, 1)
    bc.apply(A)
    bc.apply(L)

    U = Function(V)
    solve(A, U.vector(), L)

    # plot
    plt.figure()
    p = plot(U)
    plt.show()

def test_2():
    # test_2
    # constructs the mesh on a true annulus

    # -------------------------------------------------------------------
    # define the domain; "square annulus"
    eps = -0.0
    a = 1.0 + eps
    b = 2.0 + eps
    domain_outside = Rectangle(Point(-b,-b), Point(b,b))
    domain_inside = Rectangle(Point(-a,-a), Point(a,a))
    domain = domain_outside - domain_inside
    mesh = generate_mesh(domain, 100)

    class OnBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    OnBoundary().mark(boundaries, 1)

    # notice, you can see that the mesh completely excludes the centre of the annulus
    # plt.figure()
    # plot(mesh)
    # plt.show()

    # -------------------------------------------------------------------
    # define function spaces 

    V_1 = FunctionSpace(mesh, "CG", 1)
    V = BlockFunctionSpace([V_1])

    u = BlockTrialFunction(V)[0]
    v = BlockTestFunction(V)[0]

    a = [[inner(grad(u), grad(v))*dx + u*v*dx]]
    l = [v*dx]

    A = block_assemble(a)
    L = block_assemble(l)

    bc_ = DirichletBC(V.sub(0), Constant(0.), boundaries, 1)
    bc = BlockDirichletBC([bc_])
    bc.apply(A)
    bc.apply(L)

    U = BlockFunction(V)
    block_solve(A, U.block_vector(), L)

    # plot
    plt.figure()
    p = plot(U[0])
    plt.show()

def test_3():
    # domain IDs
    outside_dom = 1
    inside_dom = 2

    eps = -0.0
    a = 1.0 + eps
    b = 2.0 + eps
    domain = Rectangle(Point(-b,-b), Point(b,b))
    domain_inside = Rectangle(Point(-a,-a), Point(a,a))
    # domain = domain_outside - domain_inside
    domain.set_subdomain(outside_dom, domain)
    domain.set_subdomain(inside_dom, domain_inside)
    mesh = generate_mesh(domain, 150)

    # plt.figure()
    # plot(mesh)
    # plt.show()

    class OnBoundary(SubDomain): # outside boundary
        def inside(self, x, on_boundary):
            return on_boundary or ((near(x[0], -a) or near(x[0], a)) and between(x[1], [-a, a]) or (near(x[1], -a) or near(x[1], a)) and between(x[0], [-a, a]))
    # class OnInterface(SubDomain): # inside boundary
    #     def inside(self, x, on_boundary):
    #         return (near(x[0], -a) or near(x[0], a)) and between(x[1], [-a, a]) or (near(x[1], -a) or near(x[1], a)) and between(x[0], [-a, a])

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    on_boundary = OnBoundary()
    on_boundary.mark(boundaries, 1)
    # on_interface = OnInterface()
    # on_interface.mark(boundaries, 2)

    nudge = 0.00 # look into getting rid of this...
    class Inside(SubDomain):
        def inside(self, x, on_boundary):
            return between(x[0],[-a,a]) and between(x[1],[-a,a])
    class Outside(SubDomain):
        def inside(self, x, on_boundary):
            return not(between(x[0],[-a+nudge,a-nudge]) and between(x[1],[-a+nudge,a-nudge]))

    # create subdomains
    subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains()) # 4th argument passes the domain tags to the mesh function

    # restrictions
    # boundary_restriction = MeshRestriction(mesh, on_boundary)
    # interface_restriction = MeshRestriction(mesh, on_interface)
    outside = Outside()
    outside_restriction = MeshRestriction(mesh, outside)
    # inside = Inside()
    # inside_restriction = MeshRestriction(mesh, inside)

    # -------------------------------------------------------------------
    # define function spaces 

    V_1 = FunctionSpace(mesh, "CG", 1)
    V = BlockFunctionSpace([V_1], restrict=[outside_restriction])

    u = BlockTrialFunction(V)[0]
    v = BlockTestFunction(V)[0]

    dx = Measure("dx")(subdomain_data=subdomains)
    dx = dx(outside_dom)
    # ds = Measure("ds")(subdomain_data=boundaries)
    # dS = Measure("dS")(subdomain_data=boundaries)
    # dS = dS(2) # restrict to the interface, which has facet ID equal to 2


    a = [[inner(grad(u), grad(v))*dx + u*v*dx]]
    l = [v*dx]

    A = block_assemble(a)
    L = block_assemble(l)

    bc_ = DirichletBC(V.sub(0), Constant(0.), boundaries, 1)
    bc = BlockDirichletBC([bc_])
    bc.apply(A)
    bc.apply(L)

    U = BlockFunction(V)
    block_solve(A, U.block_vector(), L)

    # plot
    plt.figure()
    p = plot(U[0])
    plt.show()




test_3()