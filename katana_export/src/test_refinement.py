from fenics import *
from multiphenics import *
from mshr import *
import matplotlib.pyplot as plt
parameters["refinement_algorithm"] = "plaza_with_parent_facets"

# define the domain; "square annulus"
pert = 0
h_max = 10

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
mesh = generate_mesh(domain, h_max)
mesh = mesh

# class OnBoundary(SubDomain):
#     def inside(self, x, on_boundary):
#         return on_boundary
class OnInterface(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], -a) or near(x[0], a)) and between(x[1], [-a, a]) or (near(x[1], -a) or near(x[1], a)) and between(x[0], [-a, a])

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
# on_boundary = OnBoundary()
# on_boundary.mark(boundaries, 1)
on_interface = OnInterface()
on_interface.mark(boundaries, 2)


# refinement
nor = 3
for i in range(nor):
    edge_markers = MeshFunction("bool", mesh, mesh.topology().dim()-1,False)
    on_interface.mark(edge_markers, True)
    mesh = refine(mesh,edge_markers)
    # mesh = mesh.child()


plt.figure()
plot(mesh)
plt.show()