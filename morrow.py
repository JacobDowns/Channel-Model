from numpy import * 
from dolfin import *
from adams_solver import *
from channel_tools import *

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



#### Constants ###          
# Seconds per day
spd = 60.0 * 60.0 * 24.0
# Seconds in a year
spy = spd * 365.0                            
# Density of water (kg / m^3)
rho_w = 1000.0  
# Density of ice (kg / m^3)
rho_i = 910.0
# Gravitational acceleration (m / s^2)
g = 9.81 
# Flow rate factor of ice (1 / Pa^3 * s) 
A = 5.0e-25
# Average bump height (m)
h_r = 0.1
# Typical spacing between bumps (m)
l_r = 2.0
# Sheet width under channel (m)
l_c = 2.0          
# Sheet conductivity (m^(7/4) / kg^(1/2))
k = 1e-2
# Channel conductivity (m^(7/4) / kg^(1/2))
k_c = 1e-1
# Specific heat capacity of ice (J / (kg * K))
c_w = 4.22e3
# Pressure melting coefficient (J / (kg * K))
c_t = 7.5e-8
# Latent heat (J / kg)
L = 3.34e5
# Exponents 
alpha = 5. / 4.
beta = 3. / 2.
delta = beta - 2.0
# Regularization parameters for convergence
phi_reg = 1e-10
N_reg = 1e-10



### Define the basal sliding speed and melt rate. ###
class Melt(Expression):
  def eval(self,value,x):
    #value[0] = max((0.14 - sqrt(x[0] * h_max**2 / length) * 1e-4) / spd, 0.0)
    value[0] = (2.0 * (1.0 - (x[0] / length)) + 0.1) / spy 
        
class Sliding(Expression):
  def eval(self,value,x):
    value[0] = 30.0 / spy
    
# Basal sliding speed (m / s)
u_b = project(Sliding(), V)
# Melt rate (m / s)
m = project(Melt(), V)




### Set up the sheet model ###

# Unknown sheet thickness defined on continuous domain
h = Function(V)
h.interpolate(Constant(0.05))
# Sheet thickness on the channel edges
h_e = Function(V_edge)
# Unknown potential
phi = Function(V)
# Potential at the previous time step
phi_prev = Function(V)
# Ice overburden pressure
p_i = project(rho_i * g * H, V)
# Potential due to bed slope
phi_m = project(rho_w * g * B, V)
# Driving potential
phi_0 = project(p_i + phi_m, V)
# Effective pressure
N = phi_0 - phi
# Flux vector
q = -Constant(k) * h**alpha * (dot(grad(phi), grad(phi)) + Constant(phi_reg))**(delta / 2.0) * grad(phi)
# Opening term (m / s) with a switch that turns it off if the bedrock bump 
# height has been exceeded
#w = conditional(gt(h, h_r), 0.0, u_b * (Constant(h_r) - h) / Constant(l_r))
w = u_b * (h_r - h) / Constant(l_r)
# Closing term
v = Constant(A) * h * N**3
# Water pressure
p_w = Function(V)
# Water pressure as a fraction of overburden
pfo = Function(V)
# Effective pressure
N_n = Function(V)
# Effective pressure on edges
N_e = Function(V_edge)



### Set up the channel model ###

# Channel cross sectional area defined on edges
S = Function(V_edge)
# S**alpha. This is a work around for a weird bug which causes exponentiation
# on S to fail for no apparent reason.
S_exp = Function(V_edge)
# Normal and tangent vectors 
n = FacetNormal(mesh)
t = as_vector([n[1], -n[0]])
# Derivative of phi along channel 
dphi_ds = dot(grad(phi), t)
# Discharge through channels
Q = -Constant(k_c) * S_exp * abs(dphi_ds + Constant(phi_reg))**delta * dphi_ds
# Approximate discharge of sheet in direction of channel
q_c = -Constant(k) * h**alpha * abs(dphi_ds + Constant(phi_reg))**delta * dphi_ds
# Energy dissipation 
Xi = Q * -dphi_ds + abs(Constant(l_c) * q_c * dphi_ds)
# f is a switch that turns on or off the sheet flow's contribution to refreezing 
f = conditional(gt(S, 0.0), 1.0, 0.0)
# Sensible heat exchange
Pi = -Constant(c_t * c_w + rho_w) * (Q + f * l_c * q_c) * dot(grad(phi - phi_m), t)
# Another channel source term
w_c = ((Xi - Pi) / Constant(L)) * Constant((1. / rho_i) - (1. / rho_w))
# Channel creep closure rate
v_c = Constant(A) * S * N*3
# Derivative of phi along channel defined on channel edges
dphi_ds_e = Function(V_edge)
# Derivative of the water pressure defined on channel edges
dpw_ds_e = Function(V_edge)



### Set up the PDE for the potential ###

theta = TestFunction(V)
dphi = TrialFunction(V)

# Sheet model contribution to the weak form
F_s = -dot(grad(theta), q) * dx + (w - v - m) * theta * dx
# Channel contribution to the weak form
F_c = avg(-dot(grad(theta), t) * Q + (w_c - v_c) * theta) * dS
# Variational form
F = F_s + F_c

# Define the margin boundary
def margin(x, on_boundary):
  return on_boundary and near(x[0], 0.0)
  
# Set the potential on the margin to 0 (corresponding to 0 water pressure
# and 0 bed slope). Everywhere else will have homogeneous Neumann bc's that
# prevent inflow
bc = DirichletBC(V, 0.0, margin)

# Compute Jacobian
J = derivative(F, phi, dphi)



### Set up the sheet / channel size ODE ### 

# A handy object that Doug wrote for computing the slopes on channel edges
# as well as finding the midpoint value of continuous functions along channels
edge_project = EdgeProjector(V, V_edge, mesh)

# This is another nifty trick stolen from Doug. This just generates a mask
# that will be used to restrict the channel opening closing temrms to apply
# everywhere except the boundaries of the domain
theta_e = TestFunction(V_edge)
mask = assemble(theta_e('+') * dS)
mask[mask.array() > 0.0] = 1.0
mask = mask.array()

# This function derives several useful values from the potential including
# values necessary for solving the ODE
def derive_values():
  # Derive effective pressure
  N_n.vector()[:] = phi_0.vector().array() - phi.vector().array()
  # Get the effective pressure on the edges
  edge_project.midpoint(N_n, N_e)
  # Compute derivative of potential along channels
  edge_project.ds(phi, dphi_ds_e)
  # Derive the water pressure
  p_w.vector()[:] = phi.vector().array() - phi_m.vector().array()
  # Water pressure as a fraction of overburden
  pfo.vector()[:] = p_w.vector().array() / p_i.vector().array()
  # Compute the derivative of water pressure along the channels
  edge_project.ds(p_w, dpw_ds_e)  
  # Sheet thickness on channel edges. This is technically an unknown in the ODE
  # but I'm going to leave it fixed for now
  edge_project.midpoint(h, h_e)

# Slope function for the sheet
def f_h(t, Xs) :
  # Get the sheet height
  h_n = Xs[0]
  # Update global variables
  h.vector()[:] = h_n
  # Sheet stiffness matrix
  K_0 = assemble((w-v)*theta*dx)
  # Sheet mass matrix
  M_0 = assemble(theta*dx)
  dh_dt = K_0.array() / M_0.array()
  return dh_dt
  

# Slope function for the channel
def f_S(t, Xs):
  # Get the channel area
  S_n = Xs[1]
  
  # Ensure that the channel area is positive
  S_n[S_n < 0.0] = 0.0
  
   # Get effective pressures, sheet thickness on edges.
  N_n = N_e.vector().array()
  h_n = h_e.vector().array()
  
  # Array form of the derivative of the potential 
  phi_s = dphi_ds_e.vector().array()  
  # turn off pressure melting if S<=0
  fl = S_n > 0

  # Along channel flux
  Q_n = -k_c * S_n**alpha * (phi_s**2 + phi_reg)**((beta - 2.) / 2.) * phi_s
  # Flux of sheet under channel
  q_n = -k * h_n * (phi_s**2 + phi_reg)**((beta - 2.) / 2.) * phi_s
  
  # Dissipation melting due to turbulent flux
  Xi_n = abs(Q_n * phi_s) + abs(l_c * q_n * phi_s)

  # Melting due to change in PMP
  Pi_n = 0.3*(Q_n + fl * l_c * q_n) * phi_s

  # Creep closure
  v_c_n = A * S_n * (N_n**2 + N_reg) * N_n

  # Total opening rate
  opening = (Xi_n - Pi_n) / (rho_i * L)
  # Dissalow negative opening rate where the channel area is 0
  #opening[opening[S_n == 0.0] < 0.0] = 0.0

  # Calculate rate of channel size change
  dSdt = mask * (opening - v_c_n)

  return dSdt


### Set up the simulation ###

# Set some parameters for the Newton solver
params = NonlinearVariationalSolver.default_parameters()

params['newton_solver']['relaxation_parameter'] = 1.0
params['newton_solver']['relative_tolerance'] = 1e-5
params['newton_solver']['absolute_tolerance'] = 1e-5
params['newton_solver']['error_on_nonconvergence'] = True
params['newton_solver']['maximum_iterations'] = 30
params['newton_solver']['linear_solver'] = 'umfpack'

# Simulation end time
T = 100.0 * spd
# Maximum time step
dt_max = 60.0 * 60.0 * 6.0
# Initial time step
dt = 60.0 * 60.0 * 6.0

# Create a solver object
#ode_solver = RKSolve([h, S], [f_h, f_S], dt = dt, dt_max = dt_max, tol = 1e-5)
ode_solver = AdamsSolver([h, S], [f_h, f_S], init_t = 0.0, init_dt = dt, dt_max = dt_max, tol = 1e-6, verbose = True)  


# Create output files 
out_dir = "morrow_results/"
out_h = File(out_dir + "h.pvd")
out_h_e = File(out_dir + "h_e.pvd")
out_phi = File(out_dir + "phi.pvd")
out_pfo = File(out_dir + "pfo.pvd")
out_N = File(out_dir + "N.pvd")
out_N_e = File(out_dir + "N_e.pvd")
out_dphi_ds = File(out_dir + "dphi_ds.pvd")
out_pfo = File(out_dir + "pfo.pvd")
out_S = File(out_dir + "S.pvd")
out_q = File(out_dir + "q.pvd")
out_S_exp = File(out_dir + "S_exp.pvd")

# Output some of the static functions as well
File(out_dir + "B.pvd") << B
File(out_dir + "H.pvd") << H
File(out_dir + "p_i.pvd") << p_i
File(out_dir + "phi_m.pvd") << phi_m
File(out_dir + "phi_0.pvd") << phi_0
File(out_dir + "m.pvd") << m

# Create some facet functions to display functions defined on channel edges
S_f = FacetFunction('double',mesh)
S_exp_f = FacetFunction('double',mesh)
h_f = FacetFunction('double',mesh)
dphi_ds_f = FacetFunction('double',mesh)
N_f = FacetFunction('double',mesh)

# Iteration count
i = 0



### Run the simulation ###

while ode_solver.t <= T :     
  print ("Current Time:", ode_solver.t / spd)
  
  for r in [1.0, 0.1, 0.1] :
    try:
      params['newton_solver']['relaxation_parameter'] = r
      solve(F == 0, phi, bc, J = J, solver_parameters = params)
      break
    except:
      # If the initial solve fails retry it with a different relaxation parameter
      print "Solve failed: Reducing relaxation parameter."
  
  # Derive some values from the potent   ial   
  derive_values()
    
  # Update the sheet thickness and channel size
  ode_solver.step(dt) 
  
  print ("S bounds", S.vector().min(), S.vector().max())
  print ("h bounds", h.vector().min(), h.vector().max())
  
  # Make sure that s and h are positive
  indexes = S.vector().array() < 0.0
  S.vector()[indexes] = 0.0
  
  # Compute S**alpha
  S_exp.vector()[:] = S.vector().array()**alpha
  
  if i % 1 == 0:
    # Output a bunch of stuff
    out_h << h
    out_phi << phi
    out_pfo << pfo
    out_N << N_n
    out_q << project(q, Vv)

    
    # Copy some functions to facet functions for display purposes
    edge_project.copy_to_facet(S, S_f)
    #edge_project.copy_to_facet(h_e, h_f)
    #edge_project.copy_to_facet(dphi_ds_e, dphi_ds_f)
    #edge_project.copy_to_facet(N_e, N_f)
    #edge_project.copy_to_facet(S_exp, S_exp_f)
    
    out_S << S_f
    #out_h_e << h_f
    #out_dphi_ds << dphi_ds_f
    #out_N_e << N_f
    #out_S_exp << S_exp_f
  
  i += 1

  