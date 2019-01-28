import numpy as np

def euc_dist(node1, node2):
  return np.linalg.norm(node1['pos'] - node2['pos'])



def clamp(num, val, tol=1e-12):
  if abs(num - val) < tol: return val
  else: return num