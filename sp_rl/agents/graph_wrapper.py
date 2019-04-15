#!/usr/bin/env python
"""Weighted graph class that agent uses to represent the problem and calculate features"""

from graph_tool.all import *
import numpy as np
from itertools import islice
import time
from copy import deepcopy
import networkx as nx

class GraphWrapper(object):
  def __init__(self, graph_info):
    self.graph_info = graph_info
    self.adj_mat    = graph_info['adj_mat']
    self.pos        = graph_info['pos'] if 'pos' in graph_info else None
    self.source     = graph_info['source_node']
    self.target     = graph_info['target_node']
    self.train_edge_statuses = graph_info['train_edge_statuses']
    self.action_to_edge      = graph_info['action_to_edge']
    self.edge_to_action      = graph_info['edge_to_action']
    self.G      = self.to_graph_tool(self.adj_mat)
    self.nedges = self.G.num_edges()
    estat       = self.G.new_edge_property("int")
    self.G.edge_properties['status'] = estat
    self.Gnx = nx.from_numpy_matrix(self.adj_mat)

    # if pos is not None:
    #   vert_pos = self.G.new_vertex_property("vector<double>")
    #   self.G.vp['pos'] = vert_pos
    #   for v in self.G.vertices():
    #     vert_pos[v] = pos[v]

    # if edge_priors is not None:
    #   eprior = self.G.new_edge_property("double")
    #   self.G.edge_properties['p_free'] = eprior
    if self.train_edge_statuses is not None:
      self.edge_prior_vec = np.mean(self.train_edge_statuses, axis=0)    
    for edge in self.G.edges():
      self.G.edge_properties['status'][edge] = -1
    
    self.curr_sp = self.shortest_path(self.source, self.target, 'weight', 'euc_dist')
    self.curr_sp_len = self.path_length(self.in_edge_form(self.curr_sp))
    # self.sp_iterator = all_paths(self.G, self.source, self.target)#, self.G.edge_properties['weight'], epsilon=0.1)
    tm = time.time()
    self.ksp_vec = self.ksp_centrality(num_paths=3000)
    print('ksp time = ', time.time()-tm)
    self.num_invalid_checked = 0.0
    self.num_valid_checked = 0.0
    self.total_checked = 0.0

    # print self.k_shortest_paths(self.source, self.target, 10, 'weight')
    # print self.curr_sp
    
    # self.ftr_params = ftr_params
    # self.k_sp_nodes, self.k_sp, self.k_sp_len = self.k_shortest_paths(ftr_params['k'], attr='weight')
    # self.k_sp_nodes_f = deepcopy(self.k_sp_nodes)
    # self.k_sp_f = deepcopy(self.k_sp)
    # self.k_sp_len_f = deepcopy(self.k_sp_len)
    # print ('Time taken for k shortest paths = {}'.format(time.time()-start_t))
    # nx.set_edge_attributes(self.G, 0.0, 'k_short_num')
    # nx.set_edge_attributes(self.G, -1.0, 'k_short_len')
    # for sp, l in zip(self.k_sp, self.k_sp_len):
      # for e in sp:
        # self.G[e[0]][e[1]]['k_short_num'] = self.G[e[0]][e[1]]['k_short_num'] + 1.0
        # if self.G[e[0]][e[1]]['k_short_len'] == -1.0:
          # self.G[e[0]][e[1]]['k_short_len'] = l
        # else: self.G[e[0]][e[1]]['k_short_len'] = self.G[e[0]][e[1]]['k_short_len'] + l
        # self.G[e[0]][e[1]]['k_short_eval'] = 0.0   


  def reset(self):
    # self.G = self.to_graph_tool(self.adj_mat) #Graph(directed=False)
    # estat = self.G.new_edge_property("int")
    # self.G.edge_properties['status'] = estat
    
    # if self.pos is not None:
    #   vert_pos = self.G.new_vertex_property("vector<double>")
    #   self.G.vp['pos'] = vert_pos
    #   for v in self.G.vertices():
    #     vert_pos[v] = self.pos[v]
        
    # if self.edge_priors is not None:
    #   eprior = self.G.new_edge_property("double")
    #   self.G.edge_properties['p_free'] = eprior
    
    for edge in self.G.edges():
      self.G.edge_properties['status'][edge] = -1
      self.G.edge_properties['weight'][edge] = self.adj_mat[int(edge.source()), int(edge.target())]
      # if self.edge_priors is not None:
      #   eprior[edge] = self.edge_priors[(edge.source(), edge.target())]

    

    self.num_invalid_checked = 0.0
    self.num_valid_checked = 0.0
    self.total_checked = 0.0
    self.curr_sp     = self.shortest_path(self.source, self.target, 'weight', 'euc_dist')
    self.curr_sp_len = self.path_length(self.in_edge_form(self.curr_sp))
    
    # if not self.lite_ftrs:
    #   self.k_sp_nodes = deepcopy(self.k_sp_nodes_f)
    #   self.k_sp       = deepcopy(self.k_sp_f)
    #   self.k_sp_len   = deepcopy(self.k_sp_len_f)
    #   nx.set_edge_attributes(self.G, 0.0, 'k_short_num')
    #   nx.set_edge_attributes(self.G, -1.0, 'k_short_len')
    #   for sp, l in zip(self.k_sp, self.k_sp_len):
    #     for e in sp:
    #       self.G[e[0]][e[1]]['k_short_num'] = self.G[e[0]][e[1]]['k_short_num'] + 1.0
    #       if self.G[e[0]][e[1]]['k_short_len'] == -1.0:
    #         self.G[e[0]][e[1]]['k_short_len'] = l
    #       else: self.G[e[0]][e[1]]['k_short_len'] = self.G[e[0]][e[1]]['k_short_len'] + l
    #       self.G[e[0]][e[1]]['k_short_eval'] = 0.0   

  def k_shortest_paths(self, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(self.Gnx, source, target, weight=weight), k))

  def shortest_path(self, source, target, attr='weight', dist_fn = 'euc_dist'):
    sp, _ = shortest_path(self.G, self.G.vertex(source), self.G.vertex(target), self.G.edge_properties['weight'])  
    return sp
      
  def update_edge(self, eid, new_status, prev_status=-1):
    """Update the status of an edge in the graph
    Params:
      eid (int): Edge id
      new_status (int): -1 (unknown), 0 (known invalid), 1(known valid) 
      prev_status (int): -1 (unknown), 0 (known invalid), 1(known valid) 
    """
    edge = self.action_to_edge[eid]
    gt_edge = self.G.edge(edge[0], edge[1])
    self.G.edge_properties['status'][gt_edge] = new_status 
    if new_status == 0: 
      self.G.edge_properties['weight'][gt_edge] = np.inf #if invalid set weight to infinity
      if prev_status == -1:
        self.num_invalid_checked += 1.0
        self.total_checked += 1.0 
      self.curr_sp = self.shortest_path(self.source, self.target, 'weight', 'euc_dist') #recalculate shortest path if edge invalid
      self.curr_sp_len = self.path_length(self.in_edge_form(self.curr_sp))
    
    elif new_status == -1: 
      self.G.edge_properties['weight'][gt_edge] = self.adj_mat[int(edge[0]), int(edge[1])]
      if prev_status == 0:
        self.num_invalid_checked += 1.0
        self.total_checked += 1.0
      if prev_status == 1:
        self.num_valid_checked +=1.0
        self.total_checked +=1.0
      self.curr_sp = self.shortest_path(self.source, self.target, 'weight', 'euc_dist') #recalculate shortest path if edge invalid
      self.curr_sp_len = self.path_length(self.in_edge_form(self.curr_sp))
        
    elif new_status == 1:
      self.num_valid_checked += 1.0
      self.total_checked += 1.0

  def ksp_centrality(self, num_paths=100):
    ksp_vec = np.zeros(self.nedges) 
    ksp = self.k_shortest_paths(self.source, self.target, num_paths, 'weight')
    for sp in ksp:
      # sp_edge = self.in_edge_form(sp)
      for i in xrange(len(sp)-1):
        edge = (sp[i], sp[i+1])
        # print edge, self.edge_to_action[edge]
        ksp_vec[self.edge_to_action[edge]] += 1.0
    return ksp_vec/num_paths*1.0



  def get_features(self, eids, obs, itr):
    """Calculate features for a given set of edges
    Params:
      eids(array of ints)
      obs (array of ints)
      itr(int)
    Returns:
      features (array of floats)
    """
    priors         = self.edge_prior_vec[eids]
    posterior      = self.get_posterior(eids, obs)
    forward_scores = 1.0 - np.arange(eids.size())/eids.size()
    delta_lens     = self.get_delta_len_util(eids)
    delta_progs    = self.get_delta_prog_util(eids)

    features = np.concatenate(forward_scores, priors, posteriors, delta_lens, delta_progs)
    return features

  def get_priors(self, idxs):
    """Calculate prior probability for edge being in collision
    Params:
      idxs: array of edge ids to return prior over
    """
    priors = self.edge_prior_vec[idxs]
    return priors

  def get_posterior(self, idxs, obs, T=1.0):
    """Posterior over edges using current uncovered world to calculate probability of a training world
       to be the real one.
       train_edges: N X E binary matrix where N is number of worlds and E is number of edges.
       obs: 1XE with 0 for known invalid, 1 for known valid and -1 for unknown.
       T: Temperature of the distribution.
    """
    obs_idxs = np.where(obs >= 0)[0] #get idxs of edges that have been checked
    obs_checked = obs[obs_idxs]
    train_edges_checked = self.train_edge_statuses[:, obs_idxs]
    #Calculate deviation between training dataset and current observation  
    deviation = np.abs(train_edges_checked - obs_checked) 
    deviation = np.sum(deviation, axis=1)
    #Calculate probability of each world being true using softmax 
    scores = -1.0*deviation
    scores = scores - np.max(scores) #Add to make softmax numberically stable
    exp_scores = np.exp(scores/T)
    probs = exp_scores/np.sum(exp_scores)
    #Calculate the posterior (expected value of status of an edge)
    posterior = self.train_edge_statuses * probs.reshape(probs.shape[0],1)
    posterior = np.sum(posterior , axis=0)
    post = 1.0 - posterior[idxs]
    return post    
  
  def get_delta_len_util(self, eids):
    delta_lens = np.zeros(len(eids))
    for i, eid in enumerate(eids):
      dl = self.edge_delta_len(eid)
      # print dl, self.curr_shortest_path_len, dl + self.curr_shortest_path_len, 1.807061 
      # if self.curr_shortest_path_len + dl > 1.807061:
      delta_lens[i] = dl
    return delta_lens

  def get_delta_prog_util(self, eids):
    delta_progs = np.zeros(len(eids))
    for i, eid in enumerate(eids):
      dp = self.edge_delta_prog(eid)
      delta_progs[i] = dp
    return delta_progs

  def get_ksp_centrality(self, eids):
    ksp_centr = self.ksp_vec[eids]
    return ksp_centr


  def edge_delta_len(self, eid):
    curr_sp = self.curr_sp
    curr_sp_len = self.curr_sp_len
    self.update_edge(eid, 0, -1)
    new_sp = self.curr_sp
    delta_len = 0.0
    if len(new_sp) > 0:
      new_sp_len = self.curr_sp_len
      delta_len = new_sp_len - curr_sp_len
    self.update_edge(eid, -1, 0)
    return delta_len

  def edge_delta_prog(self, eid):
    curr_sp = self.curr_sp
    curr_sp_len = self.curr_sp_len
    self.update_edge(eid, 0, -1)
    new_sp = self.curr_sp
    delta_progress = 0.0
    if len(new_sp) > 0:
      new_sp_len = self.curr_sp_len
      for e in self.in_edge_form(new_sp):
        gt_edge = self.G.edge(e[0], e[1])
        if self.G.edge_properties['status'][gt_edge] == -1:
          delta_progress += 1.0
      delta_progress = delta_progress/(len(new_sp)-1)
    self.update_edge(eid, -1, 0)
    return delta_progress

  def to_graph_tool(self, adj):
    g = Graph(directed=False)
    edge_weights = g.new_edge_property('double')
    g.edge_properties['weight'] = edge_weights
    num_vertices = adj.shape[0]
    for i in range(0,num_vertices):
      for j in range(i+1,num_vertices):
        if adj[i,j]!=0:
          e = g.add_edge(i,j)
          edge_weights[e] = adj[i,j]
    return g

  def in_edge_form(self, path):
    """Helper function that takes a path(set of nodes) as input
    and returns a path as a set of edges""" 
    return [(i,j) for i,j in zip(path[:-1], path[1:])]

  @property
  def curr_shortest_path(self):
    return self.curr_sp
  @property
  def curr_shortest_path_len(self):
    return self.curr_sp_len

  def path_length(self, path):
    w = 0.0
    for edge in path:
      w = w + self.G.edge_properties['weight'][edge]
    return w

  def in_edge_tup_form(self, path):
    edge_path = self.in_edge_form(path)
    edge_tup_path = [(self.G.vertex_index[e[0]], self.G.vertex_index[e[1]]) for e in edge_path]
    return edge_tup_path