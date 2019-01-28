#!/usr/bin/env python
import os, sys
sys.path.insert(0, os.path.abspath('../..'))
from sp_rl.envs import NDGrid 

print('####2D Grid Tests####')
ndims = 2
ncells = [100, 100]
grid = NDGrid(ndims, ncells)

qpt = 0
neighs = grid.get_neighbors(qpt)
coords = grid.idx_to_coord(qpt)
print "ndims = %d, ncells=(%d, %d), query point %d - coordinates (%d, %d), neighbors (%d, %d) "%(ndims, ncells[0], ncells[1], qpt, coords[0], coords[1], neighs[0], neighs[1])
assert grid.coord_to_idx(coords) == qpt, "Did not get back query point using coord_to_idx " 
print ("grid.coord_to_idx(coords) %d == qpt %d Passed"%(grid.coord_to_idx(coords), qpt))

qpt = 100
neighs = grid.get_neighbors(qpt)
coords = grid.idx_to_coord(qpt)
print "ndims = %d, ncells=(%d, %d), query point %d - coordinates (%d, %d), neighbors (%d, %d) "%(ndims, ncells[0], ncells[1], qpt, coords[0], coords[1], neighs[0], neighs[1])
assert grid.coord_to_idx(coords) == qpt, "Did not get back query point using coord_to_idx "
print ("grid.coord_to_idx(coords) %d == qpt %d Passed"%(grid.coord_to_idx(coords), qpt))

qpt = 150
neighs = grid.get_neighbors(qpt)
coords = grid.idx_to_coord(qpt)
print "ndims = %d, ncells=(%d, %d), query point %d - coordinates (%d, %d), neighbors (%d, %d, %d, %d) "%(ndims, ncells[0], ncells[1], qpt, coords[0], coords[1], neighs[0], neighs[1], neighs[2], neighs[3])
assert grid.coord_to_idx(coords) == qpt, "Did not get back query point using coord_to_idx "
print ("grid.coord_to_idx(coords) %d == qpt %d Passed"%(grid.coord_to_idx(coords), qpt))

qpt = 9999
neighs = grid.get_neighbors(qpt)
coords = grid.idx_to_coord(qpt)
print "ndims = %d, ncells=(%d, %d), query point %d - coordinates (%d, %d), neighbors (%d, %d) "%(ndims, ncells[0], ncells[1], qpt, coords[0], coords[1], neighs[0], neighs[1])
assert grid.coord_to_idx(coords) == qpt, "Did not get back query point using coord_to_idx "
print ("grid.coord_to_idx(coords) %d == qpt %d Passed"%(grid.coord_to_idx(coords), qpt))


print('####6D Grid Tests####')
ndims = 6
ncells = [3, 3, 3, 3, 3, 3]
grid = NDGrid(ndims, ncells)
assert grid.total_cells == 3**6, "Total Cells incorrect"
print ("total cells %d == 3**6"%(grid.total_cells))

qpt = 0
neighs = grid.get_neighbors(qpt)
coords = grid.idx_to_coord(qpt)
print "ndims = %d, ncells=(%d, %d), query point %d - coordinates (%d, %d, %d, %d, %d, %d), neighbors (%d, %d) "%(ndims, ncells[0], ncells[1], qpt, coords[0], coords[1], coords[2], coords[3], coords[4], coords[5], neighs[0], neighs[1])
assert grid.coord_to_idx(coords) == qpt, "Did not get back query point using coord_to_idx " 
print ("grid.coord_to_idx(coords) %d == qpt %d Passed"%(grid.coord_to_idx(coords), qpt))

qpt = 100
neighs = grid.get_neighbors(qpt)
coords = grid.idx_to_coord(qpt)
print "ndims = %d, ncells=(%d, %d), query point %d - coordinates (%d, %d, %d, %d, %d, %d), neighbors (%d, %d) "%(ndims, ncells[0], ncells[1], qpt, coords[0], coords[1], coords[2], coords[3], coords[4], coords[5], neighs[0], neighs[1])
assert grid.coord_to_idx(coords) == qpt, "Did not get back query point using coord_to_idx "
print ("grid.coord_to_idx(coords) %d == qpt %d Passed"%(grid.coord_to_idx(coords), qpt))

qpt = 150
neighs = grid.get_neighbors(qpt)
coords = grid.idx_to_coord(qpt)
print "ndims = %d, ncells=(%d, %d), query point %d - coordinates (%d, %d, %d, %d, %d, %d), neighbors (%d, %d, %d, %d) "%(ndims, ncells[0], ncells[1], qpt, coords[0], coords[1], coords[2], coords[3], coords[4], coords[5], neighs[0], neighs[1], neighs[2], neighs[3])
assert grid.coord_to_idx(coords) == qpt, "Did not get back query point using coord_to_idx "
print ("grid.coord_to_idx(coords) %d == qpt %d Passed"%(grid.coord_to_idx(coords), qpt))

qpt = 728
neighs = grid.get_neighbors(qpt)
coords = grid.idx_to_coord(qpt)
print "ndims = %d, ncells=(%d, %d), query point %d - coordinates (%d, %d, %d, %d, %d, %d), neighbors (%d, %d) "%(ndims, ncells[0], ncells[1], qpt, coords[0], coords[1], coords[2], coords[3], coords[4], coords[5], neighs[0], neighs[1])
assert grid.coord_to_idx(coords) == qpt, "Did not get back query point using coord_to_idx "
print ("grid.coord_to_idx(coords) %d == qpt %d Passed"%(grid.coord_to_idx(coords), qpt))
