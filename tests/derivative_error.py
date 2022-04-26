from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# parameters['form_compiler']['no-evaluate_basis_derivatives'] = False

# mesh = UnitSquareMesh(32, 32)
# V = FunctionSpace(mesh, "Lagrange", 1)

# def boundary(x):
#     return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# u0 = Constant(0.0)
# bc = DirichletBC(V, u0, boundary)

# u = TrialFunction(V)
# v = TestFunction(V)
# f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
# g = Expression("sin(5*x[0])", degree=2)
# a = inner(grad(u), grad(v))*dx
# L = f*v*dx + g*v*ds

# u = Function(V)
# solve(a == L, u, bc)

# # -----------------------------------------------------------------------------------
# # evaluate

# x = Point(0.5,0.5)
# u(x) # works fine
# u.dx(1)(x) # "#todo: handle derivatives"

# https://fenicsproject.org/qa/8394/how-to-evaluate-higher-derivative-of-a-function-at-a-point/
mesh = UnitSquareMesh(3, 3)
V = FunctionSpace(mesh, 'CG', 1)
f = interpolate(Expression('x[1]',degree=3), V)
el = V.element()

# plt.figure()
# plot(f)
# # plot(p)
# plt.show()


# Where to evaluate
x = np.array([0.5, 0.5])

# Find the cell with point
x_point = Point(*x) 
cell_id = mesh.bounding_box_tree().compute_first_entity_collision(x_point)
cell = Cell(mesh, cell_id)
coordinate_dofs = cell.get_vertex_coordinates()

# Array for values with derivatives of all basis functions. 4 * element dim
# values = np.zeros(4*el.space_dimension(), dtype=float)
# Compute all 2nd order derivatives
values = el.evaluate_basis_derivatives_all(1, x, coordinate_dofs, cell.orientation()).reshape((-1,2)) # reshape parameters correct?
dofs = V.dofmap().cell_dofs(cell_id)
dofs = f.vector()[dofs]

print(np.sum(values[:,0]*dofs)) # d/dx
print(np.sum(values[:,1]*dofs)) # d/dy