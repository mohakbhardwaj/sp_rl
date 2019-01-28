import numpy as np
import networkx as nx


class RGG():
  def __init__(self, params):
    self.ndims  = params['ndims']
    self.nnodes = params['nnodes']
    self.r      = params['radius']
    self.pos = None
    if 'pos' in params:
      self.pos    = params['pos']
    self.lower_limits = np.array(params['lower_limits'])
    self.upper_limits = np.array(params['upper_limits'])
    self.path_resolution = params['path_resolution']
    self.graph = nx.generators.geometric.random_geometric_graph(self.nnodes, self.r, self.ndims, pos=self.pos, p=2, seed=None)#, self.pos)
    # self.graph = self.graph.to_directed()
    self.populate_edges()

  def get_successors(self, node):
    succs = []
    if node in self.graph:
      nbr_dict = self.graph[node]
      s1 = self.graph.node[node]['pos']
    else:
      raise ValueError, 'Node not in graph'
    for nbr in nbr_dict:
      s2 = self.graph.node[nbr]['pos'] 
      edge = self.interpolate(s1, s2, self.distance_bw_states(s1, s2)/self.path_resolution)#edge_attr['interp_edge']
      succs.append([nbr, edge])
    return succs

  def node_to_state(self, node):
    state = self.graph.node[node]['pos']
    return np.asarray(state)
  
  def state_to_node(self, state):
    #returns closest node in terms of L2 in O(N)
    state = np.asarray(state)
    min_d = np.inf
    closest_node = None
    for node in self.graph:
      s_n = self.graph.node[node]['pos']
      d = np.linalg.norm(s_n - state)
      if d < min_d:
        min_d = d
        closest_node = node
    return node

  def  distance_bw_states(self, s1, s2):
    return np.linalg.norm(np.asarray(s1) - np.asarray(s2))

  def populate_edges(self):
    for node in self.graph:
      s1 = self.graph.node[node]['pos']
      for nbr in self.graph[node]:
        s2 = self.graph.node[nbr]['pos']
        self.graph[node][nbr]['weight'] = self.distance_bw_states(s1, s2)
        self.graph[node][nbr]['status'] = -1

  def update_edge_weight(self, edge=(0,1), weight=0):
    self.graph[edge[0]][edge[1]]['weight'] = weight
  
  def set_edge_evaluated(self, edge, status):
    self.graph[edge[0]][edge[1]]['status'] = status

  def interpolate(self, s1, s2, num_points):
    """Returns a sequence of states between two states 
    given a motion that connects the two states.

    We use a shooting method type approach to interpolate between the two states

    @param: s1         - parent state
    @param: s2         - child state
    @params motion     - control input connecting the two states
    @params num_points - number of intermediate points (excluding the end points)
    @return edge       - sequence of states (including end points) 
    """
    edge = [tuple(s1)]
    length = self.distance_bw_states(s1, s2)
    d_l = length/num_points
    curr_len = d_l
    while True:
      #Do linear interpolation
      temp = [s1[i] + (s2[i] - s1[i])*(curr_len/length) for i in range(self.ndims)]
      #Update curr_len
      curr_len+= d_l
      if curr_len > length: break
      #Append temp to edge
      edge.append(tuple(temp))
    #Add final state to edge
    edge.append(tuple(s2))
    return tuple(edge)
  
  def get_edge(self, node1, node2):
    s1 = self.graph.node[node1]['pos']
    s2 = self.graph.node[node2]['pos']
    edge = self.interpolate(s1, s2, self.distance_bw_states(s1, s2)/self.path_resolution)
    return edge

  def get_continuous_path(self, path):
    return [self.get_edge(u, v) for u,v in zip(path[:-1], path[1:])]

  def path_length(self, path):
    return sum(self.graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))

  def visualize_graph(self, ax, width=1.0, edge_color='r', style='solid', alpha=1.0, cmap=None, arrows=True, label=None, node_size=300, node_color='r', node_shape='o', linewidths=None):
    nx.draw_networkx_edges(self.graph, self.pos, width = width, edge_color=edge_color, style=style, alpha = alpha, ax=ax, cmap = cmap, arrows = arrows, label=label)
    nx.draw_networkx_nodes(self.graph, self.pos, ax=ax, node_size=node_size, node_color=node_color, node_shape=node_shape, alpha=alpha, cmap=cmap, linewidths=linewidths)



if __name__=="__main__":
  import matplotlib.pyplot as plt
  import random
  lower_limits = [0, 0]
  upper_limits = [200, 200]
  n = 10
  rad = 50
  pos = {i: (np.random.uniform(lower_limits[0], upper_limits[0]), np.random.uniform(lower_limits[1], upper_limits[1])) for i in range(n)}
  params = {}
  params['ndims']  = 2
  params['nnodes'] =  n
  params['radius'] = rad
  params['pos']    = pos
  params['lower_limits'] = lower_limits
  params['upper_limits'] = upper_limits
  params['path_resolution'] = 10

  rgg = RGG(params)
  for node in rgg.graph:
    # print i, rgg.graph.node[i]['pos'], rgg.get_successors(i)
    print node, rgg.graph.node[node]['pos']
    # for nbr in rgg.graph[node]: 
      # print rgg.graph[node][nbr]['interp_edge']
      # print rgg.graph[nbr][node]['interp_edge']
  fig, ax = plt.subplots(1,1)
  rgg.visualize_graph(ax, arrows=False, node_size=70, edge_color='k')
  plt.show()