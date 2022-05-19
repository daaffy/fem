from fenics import *
from multiphenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

# domain IDs
solid_dom = 2
fluid_dom = 1

eps = -0.0
a = 1.0 + eps
b = 2.0 + eps
domain = Rectangle(Point(-b,-b), Point(b,b))
domain_inside = Rectangle(Point(-a,-a), Point(a,a))
# domain = domain_outside - domain_inside
domain.set_subdomain(solid_dom, domain)
domain.set_subdomain(fluid_dom, domain_inside)
mesh = generate_mesh(domain, 100)

class OnBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
class OnInterface(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], -a) or near(x[0], a)) and between(x[1], [-a, a]) or (near(x[1], -a) or near(x[1], a)) and between(x[0], [-a, a])

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

# save as .xml
File("mesh.xml") << mesh
File("subdomains.xml") << subdomains
File("boundaries.xml") << boundaries
File("interface_restriction.rtc.xml") << interface_restriction
File("solid_restriction.rtc.xml") << solid_restriction
File("fluid_restriction.rtc.xml") << fluid_restriction
