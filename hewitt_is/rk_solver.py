import numpy as np
from dolfin import MPI, mpi_comm_world
 
class RKSolver :
   
  def __init__(self, fns, fs, t = 0.0, dt = 0.05, dt_max = 0.1, tol = 1e-3, error_fn = None, output = True) :
    # Initial Time
    self.t = t
    # A list of Fenics functions (the unknowns)
    self.fns = fns
    # The slope function coorresponding to each function
    self.fs = fs
    # The tolerance used for adaptive time stepping
    self.tol = tol
    # Set the initial time step
    self.dt = dt
    # The maximum allowable time step
    self.dt_max = dt_max
    # Time coefficients for each k_i  
    self.t_cos = np.array([0., 1/4., 3/8., 12/13., 1., 1/2.])
    # A vector of concatenated solution vectors
    self.xs = np.hstack(np.array([self.fns[i].vector().array() for i in range(len(self.fns))]).flat)
    # A list containing the start index of every function in self.xs
    self.fn_indexes = np.cumsum([len(self.fns[i].vector().array()) for i in range(len(self.fns) - 1)])
    # Specifies whether to output info. such as the current time, time step, etc.
    self.output = output
    # MPI process rank
    self.MPI_rank = MPI.rank(mpi_comm_world())
    
    # The coefficients for computing each k_i
    self.k_cos = np.array([
      [],
      [1/4.],
      [3/32., 9/32.],
      [1932/2197., -7200/2197., 7296/2197.],
      [439/216., -8., 3680/513., -845/4104.],
      [-8/27., 2., -3544/2565., 1859/4104., -11/40.]
    ])
     
    # Coefficients for computing two RK solutions x1 and x2 of different orders given 
    # the k_i's
    self.x1_cos = np.array([16/135., 0., 6656/12825., 28561/56430., -9/50., 2/55.])
    self.x2_cos = np.array([25/216., 0., 1408/2565., 2197/4101., -1/5., 0.])
     
    # Function for estimating the error given two solutions x1 and x2 of different orders
    self.error_fn = self.default_error
    if error_fn != None :
      self.error_fn = error_fn
       
  # Advances the concatenated solutions in self.xs by dt
  def rk(self, dt) :
    # Current time
    t = self.t
    # Initialize an array of concatenated ks for all functions
    ks = [self.F(t, self.xs) * dt]
 
    # Compute each k_i
    for i in range(1, 6) :
      k_i = self.F(t + self.t_cos[i] * dt, self.xs + np.dot(self.k_cos[i], ks)) * dt
      ks.append(k_i)
       
    # Compute two solutions for all functions of different orders (x1 is higher order)
    x1 = self.xs + np.dot(self.x1_cos, ks)
    x2 = self.xs + np.dot(self.x2_cos, ks)
     
    return x1, x2
    
  # Global slope function. Takes in a time t and an array xs of concatenated 
  # function vectors. Returns an array of concatenated slope vectors.
  def F(self, t, xs):
    F = np.array([])
    
    # Get a list of individual function vectors. We will give each slope 
    # function a list of individual slope functions rather than the concatenated
    # vector since they're easier to work with
    Xs = self.split_xs()    
    
    for f in self.fs:
      F = np.append(F, f(t, Xs))
      
    return F
  
  # Splits up the concatenated solution self.xs into a list on individual function
  # vectors
  def split_xs(self) :
    return np.split(self.xs, self.fn_indexes)

  # Breaks up the concatenated solution vector self.xs and writes to each
  # Fenics function
  def write_fns(self) :
    Xs = self.split_xs()
    
    for i in range(len(self.fns)) :
      X = self.fns[i]
      X.vector().set_local(Xs[i])
      X.vector().apply("insert")
   
  # Compute the error given two concatenated solution functions of different
  # order
  def default_error(self, x1, x2) :
    dif_max = MPI.max(mpi_comm_world(), abs(x1 - x2).max())
    return dif_max
     
  # Step forward in time using a simple adaptive time stepping algorithm
  def step_adapt(self) :
    # Get two concatenated solutions vectors of different orders
    x1, x2 = self.rk(self.dt)
    # Estimate the error
    error = self.error_fn(x1, x2)
    
    if self.output :
      print "Current Time: " + str(self.t)
      print "Attempted dt: " + str(self.dt)
      print "Error: " + str(error)
    
    # Save the last dt, just in case our simulation wants it
    last_dt = self.dt
     
    # Recompute the time step based on the error
    s = 0.84 * (self.tol / (error + 1e-30))**(1/4.)
    self.dt = min(s * self.dt, self.dt_max)
 
    # If the error is greater than the tolerance, then reject this step
    # and try again
    if error > self.tol :
      if self.output :
        print "Time step rejected."
        print
      self.step_adapt()
    else :
      if self.output :
        print "Time step accepted."
        print
      
      # Update the concatenated solution 
      self.xs = x1
      # Update all of the Fenics functions
      self.write_fns()
      # Update the current time and last time step
      self.t += last_dt
      self.last_dt = last_dt
  
  # Steps forward by a specified dt using multiple steps to remain within the error
  # tolerance if necessary
  def step(self, dt) :
    
    # Target time we want to reach
    target = self.t + dt
    
    if self.output :
      print "Target: " + str(target)
    
    # Take as many steps as necessary to reach the target without exceeding
    # the tolerance
    while(abs(self.t - target) > 1e-10) :
      # Either use the last recommended time step or dt if it's smaller
      self.dt = min(dt, self.dt)
      # Step forward in time  
      self.step_adapt()
      # Update dt to be the new time to the target
      dt = target - self.t
    
    
