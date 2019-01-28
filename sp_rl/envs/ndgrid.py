#!/usr/bin/env python

class NDGrid(object):
  def __init__(self, ndims=2, num_cells=[2, 2]):
    self.ndims = int(ndims) #Dimension of grid
    self.num_cells = num_cells #Number of cells in each dimension
    self.d_ = []#Auxillary vector for speed up
    self.total_cells = 1
    for i in xrange(self.ndims):
      self.total_cells *= int(self.num_cells[i])
      self.d_.append(self.total_cells) 

  def get_cell(self, idx):
    raise NotImplementedError

  def get_neighbors(self, idx):
    assert idx >=0 and idx < self.total_cells, "invalid cell index" 
    neighs = []
    #Dimension 0
    c1 = int(idx - 1)
    c2 = int(idx + 1)
    if (c1 >= 0) and (int(c1/self.d_[0]) == int(idx/self.d_[0])):
      neighs.append(c1)
    if (c2 < self.total_cells) and (int(c2/self.d_[0]) == int(idx/self.d_[0])):
      neighs.append(c2)
    #dimension 1 ... (ndims-1)
    for i in xrange(1, self.ndims):
      c1 = idx - self.d_[i-1]
      c2 = idx + self.d_[i-1]
      if (c1 >= 0) and (int(c1/self.d_[i]) == int(idx/self.d_[i])):
        neighs.append(c1)
      if (c2 < self.total_cells) and (int(c2/self.d_[i]) == int(idx/self.d_[i])):
        neighs.append(c2)
    return neighs

  def idx_to_coord(self, idx):
    assert idx >=0 and idx < self.total_cells, "invalid cell index"
    idx = int(idx)
    coords = [0]*self.ndims
    coords[self.ndims-1] = int(idx/self.d_[self.ndims-2])
    aux = int(idx - coords[self.ndims-1]*self.d_[self.ndims-2])
    #for dimensions ndims-2 ... 1
    for i in reversed(range(1, self.ndims-1)):
      coords[i] = int(aux/self.d_[i-1])
      aux -= coords[i]*self.d_[i-1]
    coords[0] = aux
    return coords

  def coord_to_idx(self, coords):
    assert len(coords) == self.ndims, "Coordinate input must be of length %d"%(self.ndims)
    idx = coords[0]
    for i in xrange(1, self.ndims):
      idx += coords[i]*self.d_[i-1]
    return idx

  # def get_coords(self, idx):
  #   raise NotImplementedError

  # def get_idx(self, coords):
  #   raise NotImplementedError

 