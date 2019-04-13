#!/usr/bin/env python
import operator
import os
import scipy
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
import numpy as np
import networkx as nx
from graph_tool.all import *
from ndgrid import NDGrid
from sp_rl.utils.math_utils import euc_dist

class GraphEnv2D(gym.Env):
  metadata = {'render.modes': ['human'],
              'max_train_envs': 500,
              'num_train_envs': 200,
              'num_validation_envs': 200,
              'num_test_envs': 300}

  COLOR = {1 : 'g', #Known Free
           0 : 'r', #Known Invalid
           -1: str(0.5) #Unknown 
          }

  def __init__(self, dataset_folder, mode='train', file_idxing=1):
    self.dataset_folder = dataset_folder
    self.mode = mode
    self.file_idxing = file_idxing
    self.nnodes, self.nedges, self.G, self.adj_mat, self.pos, self.action_to_edge_tup, self.action_to_gt_edge, self.edge_tup_to_action, self.action_to_edge_id, self.edge_weights, self.vert_pos, self.Gdraw = self.initialize_graph(dataset_folder) 
    self.source_node, self.target_node, self.edge_statuses = self.initialize_world_distrib(dataset_folder)#, self.data_idxs)
    self.train_edge_statuses = self.edge_statuses[0:self.metadata['max_train_envs'], :]
    #Randomly shuffle the worlds
    self.curr_idx = 0
    self.first_reset = True
    if self.mode == "train":
      self.world_arr = np.arange(self.file_idxing, self.metadata['num_train_envs']+self.file_idxing)
      self.offset = 0
      self.max_envs = self.metadata['num_train_envs']
    elif self.mode == "validation": 
      self.offset = self.metadata['max_train_envs']
      self.world_arr = np.arange(self.offset + self.file_idxing, self.offset + self.metadata['num_validation_envs']+self.file_idxing)
      self.max_envs = self.metadata['num_validation_envs']
    elif self.mode == "test": 
      self.offset = self.metadata['max_train_envs'] + self.metadata['num_validation_envs']
      self.world_arr = np.arange(self.offset + self.file_idxing, self.offset + self.metadata['num_test_envs']+self.file_idxing)
      self.max_envs = self.metadata['num_test_envs']
    

    self.action_space = spaces.Discrete(self.nedges)    
    self.observation_space = spaces.Discrete(3**self.nedges)
    self.grid = NDGrid(self.nedges, [3]*self.nedges)
    self.render_called = False

  def initialize_graph(self, folder):
    with open(os.path.join(folder, "graph.txt")) as f:
      graph_lines = f.read().splitlines()
    with open(os.path.join(folder, "coord_set.dat")) as f:
      pos_lines = f.read().splitlines()

    nnodes = int(graph_lines[0].split()[1])
    nedges = int(int(graph_lines[1].split()[1])/2)
    action_to_edge_tup = {}
    edge_tup_to_action = {}
    action_to_edge_id  = {}
    action_to_gt_edge  = {}
    G = Graph(directed=False)
    edge_weights = G.new_edge_property("double")
    G.edge_properties['weight'] = edge_weights
    vert_pos = G.new_vertex_property("vector<double>")
    G.vp['pos'] = vert_pos
    edge_status = G.new_edge_property("int")
    G.edge_properties['status'] = edge_status
    edge_status = -1
    Gdraw = nx.Graph()
    print ('Reading graph from file')
    act_num = 0
    for i in xrange(2, len(graph_lines)):
      s = graph_lines[i].split()
      eid = int(s[0]) - 1 #edges ids in dataset start with 
      pid = int(s[1]) - 1 #parent node
      cid = int(s[2]) - 1 #child node
      w = float(s[3])     #edge weight
      if (pid, cid) not in edge_tup_to_action and (cid, pid) not in edge_tup_to_action:
        e = G.add_edge(pid, cid)
        Gdraw.add_edge(pid, cid)
        edge_weights[e] = w
        action_to_edge_tup[act_num]    = (pid, cid)
        edge_tup_to_action[(cid, pid)] = act_num
        edge_tup_to_action[(pid, cid)] = act_num
        action_to_edge_id[act_num]     = eid
        action_to_gt_edge[act_num]     = e
        act_num += 1
    

    adj_mat = adjacency(G, edge_weights).todense()
    pos = {}
    for i in xrange(len(pos_lines)):
      s = pos_lines[i].split(',')
      pos_f = (float(s[0]), float(s[1]))
      vert_pos[G.vertex(i)] = pos_f
      pos[i] = pos_f
    return nnodes, nedges, G, adj_mat, pos, action_to_edge_tup, action_to_gt_edge, edge_tup_to_action, action_to_edge_id, edge_weights, vert_pos, Gdraw


  def initialize_world_distrib(self, folder):#, idxs):
    #Loads the world distribution
    string = folder  + '/coll_check_results.dat'
    edge_stats = np.loadtxt(os.path.abspath(string), dtype=int, delimiter=',')
    edge_stats = edge_stats[:, list(self.action_to_edge_id.values())]
    with open(os.path.join(folder, "start_idx.dat")) as f: source_node = int(f.read().splitlines()[0].split()[0]) - 1
    with open(os.path.join(folder, "goal_idx.dat")) as f: target_node = int(f.read().splitlines()[0].split()[0])  - 1
    return source_node, target_node, edge_stats#, edge_stats_full
  
  def reinit_graph_status(self, edge_stats):
    for a,edge in self.action_to_gt_edge.iteritems():
      if edge_stats[a] == 0:
        self.G.edge_properties['status'][edge] = 0
        self.G.edge_properties['weight'][edge] = np.inf
      elif edge_stats[a] == 1:
        self.G.edge_properties['status'][edge] = 1
        self.G.edge_properties['weight'][edge] = self.adj_mat[int(edge.source()), int(edge.target())]
      else:
        raise ValueError
  
  # def calculate_edge_priors(self):
  #   edge_priors = {}
  #   train_only = self.edge_statuses[0:self.metadata['max_train_envs'], :]
  #   prior_vec =  np.mean(train_only, axis=0)
  #   for a,edge in self.action_to_edge_tup.iteritems():
  #     edge_priors[(edge[0], edge[1])] = prior_vec[a]
  #     edge_priors[(edge[1], edge[0])] = prior_vec[a]
  #   return edge_priors
  
  def step(self, action):
    done = False
    reward = -1
    self.obs[action] = self.curr_edge_stats[action] #result #Update the observation
    if self.eval_path(self.sp): #If the agent has discovered the shortest path, episode ends
      done = True
      reward = -1
    info = {} #Side information
    return self.obs, reward, done, info

  def reset(self, roll_back=False):
    self.obs = np.array([-1]*self.nedges)
    if roll_back:
      self.curr_idx = 0
      self.first_reset=True
  
    #Sample worlds till you sample a solvable one
    solvable = False
    if not self.first_reset:
      while not solvable:
        if self.mode == "train":
          self.world_num = self.world_arr[self.curr_idx]#np.random.choice(self.world_arr)
        else:
          self.world_num = self.world_arr[self.curr_idx]
        self.curr_idx = (self.curr_idx + 1) % self.max_envs
        self.curr_edge_stats = self.edge_statuses[self.world_num-self.file_idxing, :] #Sample a random world from the dataset
        self.reinit_graph_status(self.curr_edge_stats)
        self.sp, self.sp_edge = shortest_path(self.G, self.G.vertex(self.source_node), self.G.vertex(self.target_node), self.G.edge_properties['weight'])
        print(self.path_length(self.sp_edge))
        if len(self.sp) > 0:
          solvable=True
      # self.sp_edge = self.to_edge_path(self.sp)
      self.im_num =  self.world_num
      im_name = str(self.im_num)
      if self.file_idxing == 1: im_name = "world_" + str(self.im_num)
      self.img = np.flipud(plt.imread(os.path.join(self.dataset_folder, "environment_images/"+im_name+".png")))
    if self.render_called:
      plt.close(self.fig)
      self.render_called = False
    graph_info = {'adj_mat': self.adj_mat, 'pos': self.pos, 'train_edge_statuses':self.train_edge_statuses, 
                  'source_node':self.source_node, 'target_node':self.target_node, 'action_to_edge': self.action_to_edge_tup}
    self.first_reset = False
    return self.obs,  graph_info
  
  def render(self, mode='human', edge_widths={}, edge_colors={}, close=False):
    edge_width_list = []
    edge_color_list = []
    for edge in self.Gdraw.edges():
      i = self.edge_tup_to_action[edge]
      if i in edge_widths:
        edge_width_list.append(edge_widths[i])
      elif self.obs[i] != -1:
        edge_width_list.append(2.5)
      else:
        edge_width_list.append(0.5)

      if i in edge_colors and self.obs[i] == -1:
        edge_color_list.append(edge_colors[i])
      else:
        edge_color_list.append(self.COLOR[self.obs[i]])

    if not self.render_called:
      self.fig, self.ax = plt.subplots()
      self.fig.show()
      self.fig.canvas.draw()
      self.render_called = True
    self.ax.clear()
    self.ax.imshow(self.img, interpolation='nearest', origin='lower', extent=[0,1,0,1], cmap='gray')
    nx.draw_networkx_edges(self.Gdraw, self.pos, ax=self.ax, edge_color=edge_color_list, width=edge_width_list, alpha=0.6)
    nx.draw_networkx_nodes(self.Gdraw, self.pos, ax=self.ax, nodelist=[self.source_node, self.target_node], node_color=['b', 'g'], node_size=20)
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
    return self.action_to_edge_tup[action]

  def gt_edge_from_action(self, action):
    return self.action_to_gt_edge[action]

  def action_from_edge(self, edge):
    return self.edge_tup_to_action[edge]

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
  
  def euc_dist(self, node1, node2):
    pos1 = np.array(self.G.node[node1]['pos'])
    pos2 = np.array(self.G.node[node2]['pos'])
    return np.linalg.norm(pos1 - pos2)

  def path_length(self, path):
    w = 0.0
    for edge in path:
      w = w + self.G.edge_properties['weight'][edge]
    return w