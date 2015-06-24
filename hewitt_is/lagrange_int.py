import numpy as np

# Integrates 5th degree Lagrange polynomials
class LagrangeInt():
  
  # Computes a list of the integrals from start to end of the 5 basis 
  # functions for a Lagrange polynomial passing through the given (x, y) pairs
  def get_basis_integrals(self, xs, start, end):
    if len(xs) == 5:
      # Shift the polynomial so the first point is at the origin
      xs = np.array(xs) - start
      # Then the integration interval is :
      y = end - start
      # Compute the integrals of each of the basis functions
      basis_integrals = []
      
      for i in range(5) :
        basis_integrals.append(self.integrate_basis_function(xs, y, i))
      
      return basis_integrals
                  
  # Integrate a Lagrange basis function shifted to the origin from 0 to y
  def integrate_basis_function(self, xs, y, i):
    x_j = xs[i]
    x_ms = np.concatenate((xs[:i], xs[(i+1):]))
    
    a = x_ms[0]
    b = x_ms[1]
    c = x_ms[2]
    d = x_ms[3]
    
    r = a * b * c * d * y
    r -=  ((a * b * c + b * c * d + a * (b + c) * d) * y**2)/2.
    r += ((c * d + b * (c + d) + a * (b + c + d)) * y**3)/3.
    r -= ((a + b + c + d) * y**4)/4.
    r += (y**5)/5.
    r /= (x_j - a) * (x_j - b) * (x_j - c) * (x_j - d)
    return r
