from dolfin import *
from constants import *
from channel_tools_new import *
import numpy as np

### Define the mesh and ice sheet geometry ###

# Maximum ice thickness
h_max = 1500.
# Length of ice sheet 
length = 60e3

# 60km by 20km rectangular mesh
mesh = Mesh("sheet.xml")
# Standard continuous function space for the sheet model
V = FunctionSpace(mesh, "CG", 1)
# Vector function space for displaying the flux
Vv = VectorFunctionSpace(mesh, "CG", 1)
# CR function space with dofs defined on each edge for the channel model
V_edge = FunctionSpace(mesh, "CR", 1)

class Bed(Expression):
  def eval(self,value,x):
    value[0] = 0.0

class Thickness(Expression):
  def eval(self,value,x):
    value[0] = sqrt((x[0] + 50.0) * h_max**2 / length)

# Ice thickness
H = project(Thickness(), V)
# Bed elevation
B = project(Bed(), V)
# Potential
phi = Function(V)
File("hewitt_results/phi_0.xml") >> phi
plot(phi, interactive = True)

v2d = vertex_to_dof_map(V)

edge_project = EdgeProjector(V, V_edge, mesh)


# Compute an array that maps each edge to its upstream vertex dof as well
# as an array that maps each edge to it downstream vertex dof

print "start"
up_dofs, down_dofs = edge_project.get_upstream_and_downstream_dofs(phi)
print "end"

# The flux on each edge
Q = Function(V_edge)
Q.interpolate(Constant(1.0))

# Compute the sum of upstream fluxes

# Function to store the sum of upstream fluxes
Q_sums = Function(V)

print "start"
#print ((np.mgrid[:V.dim(),:V_edge.dim()] == up_dofs)[0] * Q.vector().array()).sum(axis = 1)
i = 0

print len(up_dofs)
print len(Q.vector())

for dof in up_dofs :
  Q1 = Q.vector().array()[i]
  print Q1
  Q_sums.vector().array()[dof] += Q1
  i += 1

print "end"

print max(Q_sums.vector().array())





