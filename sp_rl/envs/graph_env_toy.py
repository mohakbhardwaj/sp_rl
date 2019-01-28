#!/usr/bin/env python
import random
import operator
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from ndgrid import NDGrid

class GraphEnvToy(gym.Env):
  metadata = {'render.modes': ['human']}

  COLOR = {1 : 'g', #Known Free
           0 : 'r', #Known Invalid
           -1: '0.5' #Unknown 
          }

  def __init__(self, ver):
    self.nnodes = 5
    self.ver = ver
    if ver == 0:
      self.nedges = 6
    elif ver == 1:
      self.nedges = 8
    self.action_space = spaces.Discrete(self.nedges)    
    self.observation_space = spaces.Discrete(3**self.nedges)
    self.grid = NDGrid(self.nedges, [3]*self.nedges)
    self.render_called = False


  def initialize_world(self, nnodes, ver):
    adj_mat = np.zeros((nnodes, nnodes))
    adj_mat[0][1] = adj_mat[1][0] = 1.0
    adj_mat[0][2] = adj_mat[2][0] = 1.0
    adj_mat[0][3] = adj_mat[3][0] = 1.0
    adj_mat[1][4] = adj_mat[4][1] = 1.0
    adj_mat[2][4] = adj_mat[4][2] = 1.0
    adj_mat[3][4] = adj_mat[4][3] = 1.0
    pos = {0: (-1,0),
           1: (0, 1),
           2: (0, 0),
           3: (0,-1),
           4: (1, 0)}
    invalid_edges = []
    #Flip a coin. If 0, then (0,1) is invalid and (2,4) is invalid
    if ver == 0:      
      if np.random.binomial(1, 0.3) == 0:
        invalid_edges.append((0,1))
        invalid_edges.append((2,4))
      else: #(0,2) is free
        #Flip a coin again. If 0, then (1,4) is invalid and any one of [(0,2), (2,4), (0,3), (3,4)] is invalid
        if np.random.binomial(1, 0.5) == 0:
          invalid_edges.append((1,4))
          rem = [(0,2), (2,4), (0,3), (3,4)]
          invalid_edges.append(rem[np.random.randint(0, len(rem))])    
    elif ver == 1:
      adj_mat[1][2] = adj_mat[2][1] = 1.0
      adj_mat[2][3] = adj_mat[3][2] = 1.0
      if np.random.binomial(1, 0.4) == 0:
        invalid_edges.append((0,1))
        invalid_edges.append((2,4))
        invalid_edges.append((0,3))
      else:
        invalid_edges.append((1,4))
        invalid_edges.append((2,4))

    else:
      raise NotImplementedError
    return adj_mat, pos, invalid_edges
  
  def initialize_graph(self, adj_mat, invalid_edges):
    G = nx.from_numpy_matrix(adj_mat)

    source = 0
    target = adj_mat.shape[0]-1  
    action_to_edge = {}
    edge_to_action={}
    for i,e in enumerate(G.edges):
      action_to_edge[i] = e
      edge_to_action[e] = i
      edge_to_action[e[::-1]] = i
    
    for i, e in enumerate(action_to_edge.items()):
      edge = e[1]
      G[edge[0]][edge[1]]['status'] = 1
      G[edge[0]][edge[1]]['weight'] = 1
      if edge in invalid_edges:
        G[edge[0]][edge[1]]['status'] = 0
        G[edge[0]][edge[1]]['weight'] = np.inf

    return G, source, target, action_to_edge, edge_to_action
  
  def step(self, action): 
    done = False
    reward = -1

    e = self.action_to_edge[action] #Get edge corresponding to action id
    result = self.G[e[0]][e[1]]['status'] #Check edge for validity
    self.obs[action] = result #Update the observation
    self.G[e[0]][e[1]]['obs'] = result
    if self.eval_path(self.sp): #If the agent has discovered the shortest path, episode ends
      done = True
      reward = -1
    
    info = {} #Side information
    return self.obs, reward, done, info

  def reset(self):
    self.obs = [-1]*self.nedges
    self.adj_mat, self.pos, invalid_edges = self.initialize_world(self.nnodes, self.ver)
    self.G, self.source_node, self.target_node, self.action_to_edge, self.edge_to_action = self.initialize_graph(self.adj_mat, invalid_edges)
    self.sp = nx.shortest_path(self.G, self.source_node, self.target_node, 'weight') #The environment pre-calculates shortest path
    if self.render_called:
      plt.close(self.fig)
      self.render_called = False

    graph_info = {'adj_mat':self.adj_mat, 'pos': None, 'edge_priors': None, 'source_node':self.source_node, 'target_node':self.target_node}
    return self.obs,  graph_info
  
  def render(self, mode='human', edge_widths={}, edge_colors={}, close=False):
    edge_width_list = []
    edge_color_list = []
    for i in xrange(self.nedges):
      if i in edge_widths:
        edge_width_list.append(edge_widths[i])
      else:
        edge_width_list.append(1.0)
      if i in edge_colors and self.obs[i] == -1:
        edge_color_list.append(edge_colors[i])
      else:
        edge_color_list.append(self.COLOR[self.obs[i]])

    node_color = ['y']*self.nedges
    node_color[self.source_node] = 'b'
    node_color[self.target_node] = 'g'
    if not self.render_called:
      self.fig, self.ax = plt.subplots()
      self.fig.show()
      self.fig.canvas.draw()
      self.render_called = True
    self.ax.clear()
    nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, edge_color=edge_color_list, width=edge_width_list)
    nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, node_color=node_color)
    # nx.draw_networkx_labels(self.G, self.pos, ax=self.ax)
    self.fig.canvas.draw()

  def seed(self, seed=None):
    np.random.seed(seed)

  def close(self):
    if self.render_called:
      plt.close(self.fig)
      self.render_called = False
    
  def eval_path(self, path):
    path_edges = self.to_edge_path(path)
    n_e = len(path_edges)
    return sum([self.obs[i] for i in path_edges]) == n_e
  
  def edge_from_action(self, action):
    return self.action_to_edge[action]

  def action_from_edge(self, edge):
    return self.edge_to_action[edge]

  def obs_to_id(self, obs):
    obs = map(operator.add, obs, [1]*self.nedges)
    idx = self.grid.coord_to_idx(obs)
    return idx

  def idx_to_obs(self, idx):
    obs = self.grid.idx_to_coord(idx)
    obs = map(operator.sub, obs, [1]*self.nedges)
    return obs

  def to_edge_path(self, node_path):
    edge_path=[]
    for u, v in zip(node_path[:-1], node_path[1:]):
      edge_path.append(self.action_from_edge((u,v)))
    return edge_path

