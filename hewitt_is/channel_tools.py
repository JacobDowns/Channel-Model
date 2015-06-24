# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:07:37 2015

@author: jake
"""
from dolfin import *
import numpy as np

class EdgeProjector(object):
  def __init__(self, V, V_edge, mesh) :
    # Facet lengths
    self.facet_length = dolfin.FacetFunction('double', mesh)
    
    # First vertex index associated with each facet
    self.f_0 = dolfin.FacetFunction('uint', mesh)    
    
    # Second vertex index associated with each facet
    self.f_1 = dolfin.FacetFunction('uint', mesh)

    # Map from vertex index to degree of freedom
    self.v2d = vertex_to_dof_map(V)
    
    # Map from edge to facet
    self.e2f = self.calculate_edge_to_facet_map(V_edge)
    
    # This is an array of 1's, and -1's that help define the direction of the
    # downstream vector s
    self.s_direction = None
    
    # Calculate length of each facet and associate each facet with its
    # two vertices
    for f in dolfin.facets(mesh):
      # Vertexes associated with this facet
      v0, v1 = f.entities(0)
      
      # Compute the length of the facet
      self.facet_length[f] = np.sqrt(sum((mesh.coordinates()[v0] - mesh.coordinates()[v1])**2))
      
      # The first dof associated with this facet
      self.f_0[f] = self.v2d[v0]
      
      # The second dof associated with this facet
      self.f_1[f] = self.v2d[v1]


  # Compute the absolute value of the derivative of a function f and output it to f_out
  def ds(self, f, f_out):
    # Compute the absolute value of the slope of f across each facet
    f_out.vector()[:] = np.absolute(((f.vector().array()[self.f_1.array()] - f.vector().array()[self.f_0.array()]) / self.facet_length.array()))[self.e2f]
  
  # Basically that this is a "downstream" derivative, meaning it's always <= 0.
  def ds_down(self, f, f_out):
    f_out.vector()[:] = self.s_direction * ((f.vector().array()[self.f_1.array()] - f.vector().array()[self.f_0.array()]) / self.facet_length.array())[self.e2f]
  
  # Compute the midpoint average of a continuous function over a facet and output it to f_out   
  def midpoint(self,f, f_out):
    # Calculate midpoint average
    f_out.vector()[:] = ((f.vector().array()[self.f_1.array()] + f.vector().array()[self.f_0.array()])/2.0)[self.e2f]
    
   # This does the same as the above except it takes in the array from a continuous function
  def midpoint_array(self, f):
    # Calculate midpoint average
    return ((f[self.f_1.array()] + f[self.f_0.array()])/2.0)[self.e2f]    
  
  # This function computes the sign of each derivative phi which is used to establish
  # the downstream direction
  def set_s_direction(self, phi):
    # Computes the sign of each derivative 
    self.s_direction = -np.sign(((phi.vector().array()[self.f_1.array()] - phi.vector().array()[self.f_0.array()]) / self.facet_length.array()))[self.e2f]
    
  # This calculates the mapping from facet dof indices to facets.  It is
  # analogous to the V.dofmap().dof_to_vertex_map(mesh) method.
  def calculate_edge_to_facet_map(self, V):
    mesh = V.mesh()
    n_V = V.dim()

    # Find coordinates of dofs and put into array with index
    coords_V = np.hstack((np.reshape(V.dofmap().tabulate_all_coordinates(mesh),(n_V,2)), np.zeros((n_V,1))))
    coords_V[:,2] = range(n_V)

    # Find coordinates of facets and put into array with index
    coords_f = np.zeros((n_V,3))
    for f in dolfin.facets(mesh):
        coords_f[f.index(),0] = f.midpoint().x()
        coords_f[f.index(),1] = f.midpoint().y()
        coords_f[f.index(),2] = f.index() 

    # Sort these the same way
    coords_V = np.array(sorted(coords_V,key=tuple))
    coords_f = np.array(sorted(coords_f,key=tuple))

    # the order of the indices becomes the map
    V2fmapping = np.zeros((n_V,2))
    V2fmapping[:,0] = coords_V[:,2]
    V2fmapping[:,1] = coords_f[:,2]

    return (V2fmapping[V2fmapping[:,0].argsort()][:,1]).astype('int')
  
  
  # Takes in a function defined on a 2D CR space and copies it into a facet
  # function for visualization purposes
  def copy_to_facet(self, f, f_out) :
    f_out.array()[self.e2f] = f.vector()
    
  def copy_vector_to_facet(self, v, f_out) :
    f_out.array()[self.e2f] = v

  # Takes in a facet function f and copies it to a function in a CR space  
  def copy_facet_to_function(self, f, f_out) :
    f_out.vector()[:] = f.array()[self.e2f] 
    
