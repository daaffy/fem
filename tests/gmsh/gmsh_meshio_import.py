# from __future__ import print_function
from fenics import *
from dolfin import MeshValueCollection
import matplotlib.pyplot as plt
import meshio

# exec(open("gmsh_import.py").read())

# import mesh 
msh = meshio.read("t1.msh")

# below does not seem to work for 2d meshes
# for cell in msh.cells:
#     if cell.type == "triangle":
#         triangle_cells = cell.data # extract 2d triangle data, only one CellBlock with "triangle" type
# meshio.write("mesh.xdmf", meshio.Mesh(points=msh.points, cells={"triangle": triangle_cells})) # convert to .xdmf

# http://jsdokken.com/converted_files/tutorial_pygmsh.html
def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh

line_mesh = create_mesh(msh, "line", prune_z=True)
meshio.write("facet_mesh.xdmf", line_mesh) # boundary data

triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
meshio.write("mesh.xdmf", triangle_mesh)

mesh = Mesh()
with XDMFFile("mesh.xdmf") as infile:
    infile.read(mesh)

mvc = MeshValueCollection("size_t", mesh, 1)
with XDMFFile("facet_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

# Create mesh and define function space
# mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'P', 1)

# plt.figure()
# plot(mesh)
# plt.show()

# Define boundary condition
u_D = Expression('1', degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution and mesh
plot(u)
plot(mesh)

# Save solution to file in VTK format
# vtkfile = File('poisson/solution.pvd')
# vtkfile << u

# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
import numpy as np
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)

# Hold plot
plt.show()