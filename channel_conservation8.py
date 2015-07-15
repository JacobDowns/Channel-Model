from dolfin import *
from constants import *
from channel_tools_new import *
import numpy as np
from adams_solver import *

### Load the mesh and potential 

# Length of ice sheet 
length = 60e3

# 60km by 20km rectangular mesh
mesh = Mesh("sheet.xml")
# Standard continuous function space for the sheet model
V = FunctionSpace(mesh, "CG", 1)
# CR function space with dofs defined on each edge for the channel model
V_edge = FunctionSpace(mesh, "CR", 1)

# Load the overburden potential 
phi = Function(V)
File("hewitt_results/phi_m.xml") >> phi

# Object for dealing with the channel edges
edge_project = EdgeProjector(V, V_edge, mesh)

# Define a facet function for enforcing boundary conditions and such

# Divide
class Divide(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[0], length)
    
# Margin
class Margin(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[0], 0.0)

divide = Divide()
margin = Margin()

boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
margin.mark(boundaries, 1)
divide.mark(boundaries, 2)
ds = Measure('ds')[boundaries]

# Boundary condition for forcing 
bc = DirichletBC(V, 0.5, boundaries, 2)



### Set up the channel conservation equation

# Seconds per day
spd = 60.0 * 60.0 * 24.0
# Seconds in a year
spy = spd * 365.0                            
# Density of water (kg / m^3)
rho_w = 1000.0  
# Typical spacing between bumps (m)
l_r = 2.0        
# Sheet conductivity (m^(7/4) / kg^(1/2))
k = 1e-2
# Channel conductivity (m^(7/4) / kg^(1/2))
k_c = 1e-1
# Latent heat (J / kg)
L = 3.34e5
# Exponents 
alpha = 5. / 4.
beta = 3. / 2.
delta = beta - 2.0
# Regularization
phi_reg = 1e-10

# Channel cross-sectional area
S = Function(V_edge)
S.interpolate(Constant(0.0))
# Sheet height on the edges
h = Function(V_edge)
# Set the sheet height to an arbitrary 5cm
h.interpolate(Constant(0.05))
# Compute the derivatives of phi along the channels 
dphi_ds = Function(V_edge)
edge_project.ds(phi, dphi_ds)

# Compute an array that maps each edge to its upstream vertex dof as well
# as an array that maps each edge to it downstream vertex dof
up_dofs, down_dofs = edge_project.get_upstream_and_downstream_dofs(phi)
  
### Compute the total weight of each vertex (the sum of the derivatives of each
### downstream edge)

# Function that stores the total weight for each vertex (the sum of the weights 
# all downstream edges connected to the vertex)
total_weights = Function(V)
edges_to_up_dofs = np.array(up_dofs, dtype = int)
# An array that maps each dof to its total weight
dofs_to_total_weights = np.bincount(edges_to_up_dofs, weights = dphi_ds.vector().array())
# Not all dofs will be downstream of a channel so we might have to pad this array
pad_len = len(total_weights.vector()) - len(dofs_to_total_weights)
dofs_to_total_weights = np.append(dofs_to_total_weights, np.zeros(pad_len))
# Store the weights in a Fenics function
total_weights.vector()[:] = dofs_to_total_weights
# Output
File("total_weights.pvd") << total_weights



### Compute the normalized weight of each edge (the derivative of the edge 
### divided by the total weight of the upstream vertex)

# Facet function for plotting stuff
out_f = FacetFunction('double', mesh)

# A function that associates each edge with its upstream total weight
edges_to_weights = Function(V_edge)
edges_to_weights.vector()[:] = total_weights.vector()[up_dofs]
# Prevent division by 0
edges_to_weights.vector()[edges_to_weights.vector() == 0.0] = 1.0
# A function that that associates each edge with its normalized weight
normal_weights = Function(V_edge)
normal_weights.vector()[:] = dphi_ds.vector().array() / edges_to_weights.vector().array()
# Output to paraview
edge_project.copy_to_facet(normal_weights, out_f)
File("normal_weights.pvd") << out_f
  


### A function for computing the total upstream fluxes for each vertex and the
### upstream flux for each edge

# Function that stores the local fluxes for each edge
Q_down = Function(V_edge) 
# Function that stores the sum of upstream fluxes
Q_upstream = Function(V)
# Upstream flux associated with each edge
Q_up = Function(V_edge)
# An array that maps each edge to its downstream dof
edges_to_down_dofs = np.array(down_dofs, dtype = int)

# Computes the local and upstream fluxes for each edge
def compute_fluxes() :
  # Array form of S and h_e
  S_n = S.vector().array()
  dphi_ds_n = dphi_ds.vector().array()
  
  # Compute the local fluxes
  Q_down.vector()[:] = k_c * S_n**alpha * abs(dphi_ds_n + phi_reg)**delta * dphi_ds_n   

  # An array that maps each dof to its upstream flux (the sum of the fluxes of 
  # all upstream cahnnels connected to the vertex)
  dofs_to_upstream_fluxes = np.bincount(edges_to_down_dofs, weights = Q_down.vector().array())
  # Not all dofs will be downstream of a channel so we might have to pad the end of this array
  pad_len = len(Q_upstream.vector()) - len(dofs_to_upstream_fluxes)
  dofs_to_upstream_fluxes = np.append(dofs_to_upstream_fluxes, np.zeros(pad_len))
  # Force flux at thd divide for testing
  # Store the upstream fluxes in a Fenics function
  Q_upstream.vector()[:] = dofs_to_upstream_fluxes
  bc.apply(Q_upstream.vector())
  
  # Weight the upstream flux using the normalized channel weight
  Q_up.vector()[:] = Q_upstream.vector()[up_dofs] * normal_weights.vector().array()

  


### Set up the system of ODEs for solving the conservation equation

# Test function on CR space
v = TestFunction(V_edge)
# Compute the lengths of each edge
edge_lens = Function(V_edge)
edge_lens.vector()[:] = assemble(v('+') * dS + v * ds).array()

# RHS for method of system of ODEs
def rhs(t, Xs) :

  S_n = Xs[0]
  S_n[S_n < 0.0] = 0.0
  
  # Make sure that the S function is current
  S.vector()[:] = S_n
  
  # Compute the upstream and downstream fluxes
  compute_fluxes()
  
  # Compute the dissipation
  #Xi_n = dphi_ds.vector().array()**beta * (k_c * S.vector().array()**alpha + l_r * k * h.vector().array()**alpha)

  # Fluxes 
  dsdt = (Q_up.vector().array() - Q_down.vector().array()) / edge_lens.vector().array()
  #dsdt += Xi_n / (rho_w * L)
  
  return dsdt



ts = np.linspace(0, 146302, 1000)

dt = 60.0 * 1.0
# Create a solver object
ode_solver = AdamsSolver([S], [rhs], init_t = 0.0, init_dt = dt, dt_max = dt, tol = 1e-7, verbose = True)  

while ode_solver.t < 5.0 * spd:
  ode_solver.step(dt)
  
edge_project.copy_to_facet(S, out_f)
plot(out_f, interactive = True)

File("S_final.pvd") << out_f

quit()

i = 0
for s in sol:
  print i
  i += 1
  edge_project.copy_vector_to_facet(s, out_f)
  #plot(out_f)
  #File("out/S_final" + str(i) + ".pvd") << out_f
  
  






