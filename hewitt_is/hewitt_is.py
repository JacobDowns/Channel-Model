from channel_tools import * 
from adams_solver import *
from dolfin import *
from constants import *
from dolfin_adjoint import *
parameters['form_compiler']['precision'] = 30



### Define the mesh and ice sheet geometry ###

# Input directory
in_dir = "IS_Small/"

mesh = Mesh(in_dir + "IS_Small.xml")
# Standard continuous function space for the sheet model
V = FunctionSpace(mesh, "CG", 1)
# CR function space on edges
V_edge = FunctionSpace(mesh, "CR", 1)
# Vector function space for displaying the flux 
Vv = VectorFunctionSpace(mesh, "CG", 1)

# Bed
B = Function(V)
File(in_dir + "Bed.xml") >> B

# Thickness
H = Function(V)
File(in_dir + "H.xml") >> H
# Impose a minimum ice thickness
H = project(conditional(lt(H, 50.0), 50.0, H), V)

# Basal velocity
u_b = Function(V)
File(in_dir + "u_b.xml") >> u_b

# Melt rate
m = Function(V)
File(in_dir + "m.xml") >> m

# Facet function for marking the margin boundary
boundaries = FacetFunction("size_t", mesh)
File(in_dir + "boundaries.xml") >> boundaries



### Set up the sheet model ###

# Unknown sheet thickness defined on continuous domain
h = Function(V)
# Initialize it to a constant value
h.interpolate(Constant(0.05))
# Sheet thickness on the channel edges
h_e = Function(V_edge)
# Ice overburden pressure
p_i = project(rho_i * g * H, V)
# Potential due to bed slope
phi_m = project(rho_w * g * B, V)
# Driving potential
phi_0 = project(p_i + phi_m, V)
# Unknown potential
phi = Function(V)
# Initialize phi to a reasonable first guess - overburden pressure except at
# the margin 
phi.assign(phi_m)
#bc.apply(phi.vector())
# Effective pressure
N = phi_0 - phi
# Sheet opening rate
w = u_b * (h_r - h) / Constant(l_r)
# Closing term
v = Constant(A) * h * N**3
# Flux vector
q = -Constant(k) * h**alpha * (dot(grad(phi), grad(phi)) + Constant(phi_reg))**(delta / 2.0) * grad(phi)
# Water pressure
p_w = Function(V)
# Water pressure as a fraction of overburden
pfo = Function(V)
# Non expression form of the effective pressure
N_n = Function(V)
# Effective pressure on edges
N_e = Function(V_edge)
# Dirichlet boundary condition for enforcing 0 pressure at the margin
bc = DirichletBC(V, phi_m, boundaries, 1)



### Set up the channel model ###

# Channel cross sectional area defined on edges
S = Function(V_edge)
# S**alpha. This is a work around for a weird bug which causes exponentiation
# on S to fail for no apparent reason (probably something related to the CR space)
S_exp = Function(V_edge)
S_exp.vector()[:] = S.vector().array()**alpha
# Normal and tangent vectors 
n = FacetNormal(mesh)
t = as_vector([n[1], -n[0]])
# Derivative of phi along channel 
dphi_ds = dot(grad(phi), t)
# Discharge through channels
Q = -Constant(k_c) * S_exp * (abs(dphi_ds) + Constant(phi_reg))**delta * dphi_ds
# Channel creep closure rate
v_c = Constant(A) * S * N**3
# Derivative of phi along channel defined on channel edges
dphi_ds_e = Function(V_edge)



### Set up the functional we need to minimize to get the potential ###

# This is a guess at what phi should be. It seems the dissipation term can't be
# written as the variation of a functional so we'll just take a guess at phi to use
# in the dissipation term, minimize the functional with that fixed phi to get
# a new guess for phi, and stop when phi stops changing (basically Picard iteration)
phi_guess = Function(V)
phi_guess.assign(phi_0)
bc.apply(phi_guess.vector())

# Derivative of phi_guess along channels
dphi_ds_guess = dot(grad(phi_guess), t)
# Dissipation involging phi_guess rather than the unknown phi
Xi_guess = (Constant(k_c) * S_exp + Constant(l_r * k) * h**alpha) * abs(dphi_ds_guess)**beta 
# Sheet opening term guess
w_c_guess = Constant((rho_w - rho_i) / (rho_w * rho_i * L)) * Xi_guess

J1 = Constant((1.0 / beta) * k) * h**alpha * (dot(grad(phi), grad(phi)) + phi_reg)**(beta / 2.0)
J2 = Constant(0.25 * A) * h * N**4 
J3 = (w - m) * phi 
J4 = Constant((1.0 / beta) * k_c) * S_exp * (abs(dphi_ds) + Constant(phi_reg))**beta
J5 = Constant(0.25 * A) * S * N**4
J6 = w_c_guess * phi

# The full functional as a fenics form 
J_phi = (J1 + J2 + J3) * dx + (J4 + J5 + J6)('+') * dS

# Now we'll set up the variational problem for fenics_adjoint

# Define some upper and lower bounds for phi
phi_min = Function(V)
phi_max = Function(V)
phi_min.assign(phi_m)
phi_max.assign(phi_0)
bc.apply(phi_max.vector())

# Make a reduced functional object for dolfin_adjoint
J_func = Functional(J_phi * dt[FINISH_TIME])
reduced_functional = ReducedFunctional(J_func, Control(phi, value = phi))
# Relative tolerance for bfgs
tol = 1e-7

# A function that solves for phi 
def solve_phi():
  for k in range(5) :
    # Use the last value of phi as for phi_guess
    phi_guess.assign(phi)
    
    # Minimize the functional with the fixed phi_guess
    m_opt = minimize(reduced_functional, method = "L-BFGS-B", tol=tol, bounds = (phi_min, phi_max), options = {"disp": True})
    plot(phi)
    
    # If the new phi isn't that much different from the old phi, call it good and quit
    dif =  max(abs(phi.vector().array() - phi_guess.vector().array()))
    if dif < 10.0 :
      break



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
  # Sheet thickness on channel edges. This is technically an unknown in the ODE
  # but I'm going to leave it fixed for simplicity
  edge_project.midpoint(h, h_e)

# Slope function for the sheet
def f_h(t, Xs) :
  # Get the sheet height
  h_n = Xs[0]
  # Sheet opening term
  w_n = u_b.vector().array() * (h_r - h_n) / l_r
  # Ensure that the opening term is non-negative
  w_n[w_n < 0.0] = 0.0
  # Sheet closure term
  v_n = A * h_n * N_n.vector().array()**3
  # Return the time rate of change of the sheet
  dh_dt = w_n - v_n
  return dh_dt
  
# Slope function for the channel
def f_S(t, Xs):
  # Get the channel area
  S_n = Xs[1]
  # Get the sheet height
  h_n = Xs[0]
  # Ensure that the channel area is positive
  S_n[S_n < 0.0] = 0.0
  # Get effective pressures, sheet thickness on edges.
  N_n = N_e.vector().array()
  h_n = h_e.vector().array()
  # Array form of the derivative of the potential 
  phi_s = dphi_ds_e.vector().array()  
  # Dissipation
  Xi_n = (k_c * S_n**alpha + l_r * k * h_n**alpha) * abs(phi_s)**beta 
  # Creep closure
  v_c_n = A * S_n * N_n**3
  # Total opening rate
  v_o_n = Xi_n / (rho_i * L)
  # Calculate rate of channel size change
  dSdt = mask * (v_o_n - v_c_n)
  return dSdt



### Set up a few more things for the simulation

# Simulation end time
T = 500.0 * spd
# Maximum time step
dt_max = 60.0 * 60.0 * 6.0
# Initial time step
dt = 60.0 * 60.0 * 6.0

# Create a solver object
ode_solver = AdamsSolver([h, S], [f_h, f_S], init_t = 0.0, init_dt = dt, dt_max = dt_max, tol = 5e-7, verbose = True)  

# Create output files 
out_dir = "hewitt_results/"
out_h = File(out_dir + "h.pvd")
out_phi = File(out_dir + "phi.pvd")
out_pfo = File(out_dir + "pfo.pvd")
out_S = File(out_dir + "S.pvd")

# Output some of the static functions as well
File(out_dir + "phi_m.pvd") << phi_m
File(out_dir + "phi_0.pvd") << phi_0

# Create some facet functions to display functions defined on channel edges
S_f = FacetFunction('double',mesh)

# Iteration count
i = 0



### Run the simulation ###

while ode_solver.t <= T :     
  
  print ("Current Time:", ode_solver.t / spd)
  
  solve_phi()
  
  # Derive some values from the potent   ial   
  derive_values()
    
  # Update the sheet thickness and channel size
  ode_solver.step(dt) 
  
  # Update S**alpha
  S_exp.vector()[:] = S.vector().array()**alpha
  
  print ("S bounds", S.vector().min(), S.vector().max())
  print ("h bounds", h.vector().min(), h.vector().max())
  
  # Output to paraview
  if i % 1 == 0:
    out_h << h
    out_phi << phi
    out_pfo << pfo    
    # Copy some functions to facet functions for display purposes
    edge_project.copy_to_facet(S, S_f)
    out_S << S_f
  
  # Checkpoint
  if i % 20:
    File(out_dir + "S_" + str(i) + ".xml") << S_f
    File(out_dir + "h_" + str(i) + ".xml") << h
    File(out_dir + "pfo_" + str(i) + ".xml") << pfo
    File(out_dir + "phi_" + str(i) + ".xml") << phi

  i += 1





                                                                                                                                      









  