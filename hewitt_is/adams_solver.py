import numpy as np
from rk_solver import *
from lagrange_int import * 
from dolfin import MPI, mpi_comm_world

class AdamsSolver():
    
  def __init__(self, Ys, fs, init_t = 0.0, init_dt = 1.0, dt_max = 2.0, tol = 1e-3, verbose = False):
    # Fenics functions for each of the unknowns
    self.Ys = Ys
    # The number of unknowns
    self.num_unknowns = len(self.Ys)
    # Slope functions corresponding to each unknown
    self.fs = fs
    # Current time
    self.t = init_t
    # Initial time step
    self.dt = init_dt
    # Maximum time step
    self.dt_max = dt_max
    # Tolerance
    self.tol = tol
    # List of lists. Each sublist stores the slope functions evaluated at the 
    # last 5 solutions for the ith unknown
    self.f_ns = [[] for i in range(self.num_unknowns)]
    # List of solution vectors at last time step
    self.prev_ys = None
    # Flag for first step
    self.first = True
    # Output stuff?
    self.verbose = verbose
    # Store the last five solution times
    self.ts = []
    # An object for computing integrals of basis functions for Lagrange 
    # polynomials. This is used to determine the coefficients for the AB 
    # and AM methods given the last five solution times
    self.l_int = LagrangeInt()
    # Process rank
    self.MPI_rank = MPI.rank(mpi_comm_world())
    
  # Take 5 steps with an adaptive RK solver to bootstrap the PC method
  def bootstrap(self) :
    # Each of the steps will have a time step of 1 / 5. of the specified 
    # initial time step
    dt = self.dt / 5.0
    rk_solver = RKSolver(self.Ys, self.fs, t = self.t, dt = dt, dt_max = self.dt, tol = self.tol, output = False)
          
    if self.verbose and self.MPI_rank == 0:
      print "Spinning up with RK."

    for i in range(5) :      
      # Update the solution functions
      rk_solver.step(dt)
      # Advance time
      self.advance_time(dt)
      # Get the solutions as a list of arrays
      ys = self.get_ys(self.Ys)
      # Get the slope functions evaluated at the solutions as a list of arrays
      fs = self.get_fs(self.t, ys)
      # Store the slope vectors
      self.push_f_ns(fs)
      
      if self.verbose and self.MPI_rank == 0:
        print "RK Step " + str(i + 1) + "/5 Complete."
        print "Current Time: " + str(self.t)
        
    self.prev_ys = ys
    print
    
  # ys: a list of all solutions in array form. 
  # t: time
  # Returns an array of concatenated slope arrays
  def get_fs(self, t, ys) :
    return [self.fs[i](t, ys) for i in range(self.num_unknowns)]

  # fs : A list of slope vectors corresponding to each unknown
  # Inserts a new slope vector at the beginning of the list of slope vectors for each unknown
  def push_f_ns(self, fs):
    for i in range(self.num_unknowns):
      self.f_ns[i].insert(0, fs[i])
  
  # Pops the oldest slope vector off of the list of slope vectors for each unknown
  def pop_f_ns(self) :
    fs = []
    for i in range(self.num_unknowns):
      fs.append(self.f_ns[i].pop())
      
  # dt : time step
  # Inserts a new time step into the list of previous time steps      
  def push_ts(self, dt) :
    self.ts.insert(0, dt)
    
  # Pops the oldest dt off of the list of previous time steps
  def pop_ts(self) :
    self.ts.pop()
  
  # Ys : A list of Fenics functions
  # Returns a list of vectors corresponding to each Fenics function
  def get_ys(self, Ys) :
    return [y.vector().array() for y in Ys]

  # dt : time step
  # Advances time by dt
  def advance_time(self, dt) :
    self.t += dt
    self.push_ts(self.t)
    
    if len(self.ts) > 5:
      self.pop_ts()

  # Step solutions foward by dt using the Adams-Bashforth method and then
  # with the implicit Adams-Moulton method and return both solutions    
  def try_step(self, dt):
    # List of solution vectors at last time step
    ys0 = self.prev_ys
    # List of tentative solution vectors from predictor step
    ys1 = []
    # List of corrected solution vectors from corrector step
    ys2 = []
    
    # Determine the coefficients for the Adams-Bashforth method
    # Determine bounds of integration
    start = self.t
    end = start + dt
    ab_cs = self.l_int.get_basis_integrals(self.ts, start, end)
    
    # Advance each of the unknowns with the explicit Adams-Bashforth method
    for i in range(self.num_unknowns):
      # Compute the explicit solution
      y_hat = ys0[i] + np.dot(ab_cs, self.f_ns[i])      
      ys1.append(y_hat)
    
    # Compute the slope vectors at the tentative solution
    fs_hat = self.get_fs(end, ys1)
    
    # Determine the coefficients for the Adams-Moulton method
    ts = self.ts[:4] 
    ts.insert(0, end)
    am_cs = self.l_int.get_basis_integrals(ts, start, end)
    
    # Advance each of the unknowns with the Adams-Moulton method using the
    # predicted solution from Adams-Bashforth
    for i in range(self.num_unknowns):
      # Compute the explicit solution
      y = ys0[i] + (am_cs[0] * fs_hat[i] + np.dot(am_cs[1:], self.f_ns[i][:4]))      
      ys2.append(y)
      
    return (ys1, ys2)
  
  # y1 : A list of solutions
  # y2 : Another list of solutions 
  # Return the error between two solutions for each of the unknowns
  def error(self, ys1, ys2) :
    errors = []
    
    for i in range(self.num_unknowns):
      err = MPI.max(mpi_comm_world(), (abs(ys2[i] - ys1[i])).max())
      errors.append(err)
    
    return errors

  # Step forward in time using a simple adaptive time stepping algorithm
  def step_adapt(self):
    if self.first : 
      # For the first step, use the RK solver to initialize some stuff for the Adams solver    
      self.bootstrap()
      self.first = False
    else : 
      # Otherwise use the Adams solver
      ys1, ys2 = self.try_step(self.dt)
      # Get the errors
      errors = self.error(ys1, ys2)
      error = max(errors)
      
      if self.verbose and self.MPI_rank == 0:
        print "Start Time: " + str(self.t)
        print "Attempted dt: " + str(self.dt)
        print "Error: " + str(error)
      
      # Save the last attempted time step
      self.last_dt = self.dt
      
      # Recompute the time step based on the error
      s = 0.84 * (self.tol / (error + 1e-30))**(1/4.)
      self.dt = min(s * self.dt, self.dt_max)
   
      # If the error is greater than the tolerance, then reject this step
      # and try again
      if error > self.tol :
        if self.verbose and self.MPI_rank == 0:
          print "Time step rejected."
          print
          
        self.step_adapt()
      else :
        if self.verbose and self.MPI_rank == 0:
          print "Time step accepted."
          print
        
        # Write the new solution vectors back to their corresponding Fenics functions
        self.write_Ys(ys2)
        # Update the time
        self.advance_time(self.last_dt)
        # Compute the slope vectors at the new solution
        fs = self.get_fs(self.t, ys2)
        # Store these slope vectors
        self.push_f_ns(fs)
        # Get rid of the slope vectors corresponding to the oldest solutions
        self.pop_f_ns()
        # Store the latest solution
        self.prev_ys = ys1  
  
  # dt: time step
  # Steps forward by a specified dt using multiple steps to remain within the error
  # tolerance if necessary
  def step(self, dt) :
    # Target time we want to reach
    target = self.t + dt
    
    if self.verbose and self.MPI_rank == 0:
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
  
  # ys : list of solution vectors
  # Writes each solution vector back to its corresponding Fenics function
  def write_Ys(self, ys) :
    for i in range(self.num_unknowns) :
      Y = self.Ys[i]
      Y.vector().set_local(ys[i])
      Y.vector().apply("insert")
