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
# Pressure melting coefficient (J / (kg * K))
c_t = 7.5e-8
# Latent heat (J / kg)
L = 3.34e5
# Exponents 
alpha = 5. / 4.
beta = 3. / 2.
delta = beta - 2.0
# Regularization parameters for convergence
phi_reg = 1e-15
N_reg = 1e-15



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
w = u_b * (h_r - h) / Constant(l_r)
# Closing term
v = Constant(A) * h * N**3
# Water pressure
p_w = Function(V)
# Water pressure as a fraction of overburden
pfo = Function(V)
# Non expression form of the effective pressure
N_n = Function(V)
# Effective pressure on edges
N_e = Function(V_edge)



### Set up the channel model ###

# Channel cross sectional area defined on edges
S = Function(V_edge)
# S**alpha. This is a work around for a weird bug which causes exponentiation
# on S to fail for no apparent reason (probably something related to the CR space)
S_exp = Function(V_edge)
# Normal and tangent vectors 
n = FacetNormal(mesh)
t = as_vector([n[1], -n[0]])
# Derivative of phi along channel 
dphi_ds = dot(grad(phi), t)
# Discharge through channels
Q = -Constant(k_c) * S_exp * (abs(dphi_ds) + Constant(phi_reg))**delta * dphi_ds
# Dissipation
Xi = (Constant(k_c) * S_exp + Constant(l_r * k) * h**alpha) * abs(dphi_ds)**beta 
# Channel creep closure rate
v_c = Constant(A) * S * N**3
# Channel opening rate
w_c = Constant((rho_w - rho_i) / (rho_w * rho_i * L)) * Xi
# Derivative of phi along channel defined on channel edges
dphi_ds_e = Function(V_edge)
# Derivative of the water pressure defined on channel edges
dpw_ds_e = Function(V_edge)



### Objective function for potential ###

# Sheet model contribution to the weak form
# This is a guess at what phi should be. It seems the dissipation terms can't be
# written as the variational of a functional so we'll just take a guess at a fixed phi for
# the dissipation term, minimize the functional with that fixed phi, and use picard
# iteration to improve that guess
phi_guess = Function(V)
phi_guess.assign(phi_0)
# Derivative of phi_guess along channels
dphi_ds_guess = dot(grad(phi_guess), t)
# Dissipation involging phi_guess rather than the unknown phi
Xi_guess = (Constant(k_c) * S_exp + Constant(l_r * k) * h**alpha) * abs(dphi_ds_guess)**beta 
# Sheet opening term guess
w_c_guess = Constant((rho_w - rho_i) / (rho_w * rho_i * L)) * Xi_guess

J1 = Constant((1.0 / beta) * k) * h**alpha *(dot(grad(phi), grad(phi)) + phi_reg)**(beta / 2.0)
J2 = Constant(0.25 * A) * h * N**4 
J3 = (w - m) * phi 
J4 = Constant((1.0 / beta) * k_c) * S_exp * (abs(dphi_ds) + Constant(phi_reg))**beta
J5 = Constant(0.25 * A) * S * N**4
J6 = w_c_guess * phi
J_phi = (J1 + J2 + J3) * dx + (J4 + J5 + J6)('+') * dS

# Python function form of objective function
def J_func(x, *args):
  phi.vector()[:] = x
  return assemble(J_phi)



### Variation of the functional ###

# Function for storing the variation of the objective function 
dphi = Function(V)
# Test function
theta = TestFunction(V)
# Variation of objective function
F_s = -dot(grad(theta), q) * dx + (w - v - m) * theta * dx
F_c = -(dot(grad(theta), t) * Q + (w_c_guess - v_c) * theta)('+') * dS
F = F_s + F_c
# Jacobian

# Define the margin boundary
def margin(x, on_boundary):
  return on_boundary and near(x[0], 0.0)
  
# We need to make sure that the solution at the margin doesn't change, so 
# we'll enforce 0 variation there
bc = DirichletBC(V, 0.0, margin)

def F_func(x, *args):
  phi.vector()[:] = x
  dphi.vector()[:] = assemble(F).array()
  bc.apply(dphi.vector())
  return dphi.vector().array()



# To initialize phi_guess, solve once using the Newton solver

params = NonlinearVariationalSolver.default_parameters()
params['newton_solver']['relaxation_parameter'] = 1.0
params['newton_solver']['relative_tolerance'] = 5e-6
params['newton_solver']['absolute_tolerance'] = 5e-6
params['newton_solver']['error_on_nonconvergence'] = True
params['newton_solver']['maximum_iterations'] = 40
params['newton_solver']['linear_solver'] = 'umfpack'

solve(F == 0, phi, bc, J = J, solver_parameters = params)
phi_guess.assign(phi)
phi.assign(phi_m)

plot(phi_guess, interactive = True)
plot(phi, interactive = True)
quit()


### Solve for the potential. ###
# Define upper and lower boundaries for the pressure
bounds = zip(phi_m.vector().array(), phi_0.vector().array())

# Solve fot the potential with bfgs
def solve_phi():
  x, f, d = fmin_l_bfgs_b(J_func, phi.vector().array(), F_func, bounds = bounds)
  phi.vector()[:] = x

  






                                                                                                                                      









  