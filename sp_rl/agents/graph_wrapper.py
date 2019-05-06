#!/usr/bin/env python
"""Weighted graph class that agent uses to represent the problem and calculate features"""

from graph_tool.all import *
import numpy as np
from itertools import islice
import time
from copy import deepcopy
import networkx as nx

class GraphWrapper(object):
  def __init__(self, graph_info, k=1):
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
    self.k = k
    # if pos is not None:
    #   vert_pos = self.G.new_vertex_property("vector<double>")
    #   self.G.vp['pos'] = vert_pos
    #   for v in self.G.vertex_indexces():
    #     vert_pos[v] = pos[v]

    if self.train_edge_statuses is not None:
      self.edge_prior_vec = 1.0 - np.mean(self.train_edge_statuses, axis=0)    
    
    self.curr_sp = self.shortest_path(self.source, self.target, 'weight', 'euc_dist')
    self.curr_sp_len = self.path_length(self.curr_sp)
    self.num_invalid_checked = 0.0
    self.num_valid_checked = 0.0
    self.total_checked = 0.0

  def reset(self):   
    for edge in self.G.edges():
      self.G.edge_properties['weight'][edge] = self.adj_mat[int(edge.source()), int(edge.target())]

    self.num_invalid_checked = 0.0
    self.num_valid_checked = 0.0
    self.total_checked = 0.0
    self.curr_sp     = self.shortest_path(self.source, self.target, 'weight', 'euc_dist')
    self.curr_sp_len = self.path_length(self.curr_sp)
    
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
    # self.G.edge_properties['status'][gt_edge] = new_status 
    if new_status == 0: 
      self.G.edge_properties['weight'][gt_edge] = np.inf #if invalid set weight to infinity
      self.Gnx[edge[0]][edge[1]]['weight'] = np.inf
      if prev_status == -1:
        self.num_invalid_checked += 1.0
        self.total_checked += 1.0 
      self.curr_sp = self.shortest_path(self.source, self.target, 'weight', 'euc_dist') #recalculate shortest path if edge invalid
      self.curr_sp_len = self.path_length(self.curr_sp)
    
    elif new_status == -1: 
      self.G.edge_properties['weight'][gt_edge] = self.adj_mat[int(edge[0]), int(edge[1])]
      self.Gnx[edge[0]][edge[1]]['weight'] = self.adj_mat[int(edge[0]), int(edge[1])]
      if prev_status == 0:
        self.num_invalid_checked += 1.0
        self.total_checked += 1.0
      if prev_status == 1:
        self.num_valid_checked +=1.0
        self.total_checked +=1.0
      self.curr_sp = self.shortest_path(self.source, self.target, 'weight', 'euc_dist') #recalculate shortest path if edge invalid
      self.curr_sp_len = self.path_length(self.curr_sp)
    elif new_status == 1:
      self.num_valid_checked += 1.0
      self.total_checked += 1.0

  def ksp_centrality(self, num_paths=100):
    ksp_vec = np.zeros(self.nedges) 
    ksp = self.k_shortest_paths(self.source, self.target, num_paths, 'weight')
    for sp in ksp:
      for i in xrange(len(sp)-1):
        edge = (sp[i], sp[i+1])
        ksp_vec[self.edge_to_action[edge]] += 1.0
    return ksp_vec/num_paths*1.0



  def get_features(self, eids, obs, itr, quad=False):
    """Calculate features for a given set of edges
    Params:
      eids(array of ints)
      obs (array of ints)
      itr(int)
      quad(bool)
    Returns:
      features (array of floats)
    """
    priors         = self.get_priors(eids)
    posteriors     = self.get_posterior(eids, obs)
    forward_scores = self.get_forward_scores(eids)
    delta_lens     = self.get_delta_len_util(eids)
    delta_progs    = self.get_delta_prog_util(eids)
    # print priors.shape, posteriors.shape, forward_scores.shape, delta_lens.shape, delta_progs.shape
    features = np.concatenate((forward_scores, priors, posteriors, delta_lens, delta_progs), axis=1)
    # if quad:
    #   q = np.einsum('ij,ik->ijk', features, features)
    #   print q[0][np.triu_indices(q.shape[1], k=1)]
    #   q = q[:,np.triu_indices(q.shape[1], k=1)].reshape(features.shape[0],-1)
      
    #   features = np.concatenate((features, q), axis=1)
    return features
  
  def get_forward_scores(self, eids):
    if len(eids) > 1:
      forward_scores = 1.0 - np.arange(len(eids))*1.0/(len(eids)-1.0)
    else:
      forward_scores = np.array([1.0])
    return forward_scores.reshape(len(eids),1)


  def get_priors(self, idxs):
    """Calculate prior probability for edge being in collision
    Params:
      idxs: array of edge ids to return prior over
    """
    priors = self.edge_prior_vec[idxs]
    return priors.reshape(len(idxs),1)

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
    return post.reshape(len(idxs),1)   
  
  def get_delta_len_util(self, eids):
    delta_lens = np.zeros(len(eids))
    for i, eid in enumerate(eids):
      dl = self.edge_delta_len(eid)
      delta_lens[i] = dl
    return delta_lens.reshape(len(eids),1)

  def get_delta_prog_util(self, eids):
    delta_progs = np.zeros(len(eids))
    for i, eid in enumerate(eids):
      dp = self.edge_delta_prog(eid)
      delta_progs[i] = dp
    return delta_progs.reshape(len(eids),1)

  def get_ksp_centrality(self, eids):
    ksp_vec = self.ksp_centrality(self.k)
    ksp_centr = ksp_vec[eids]
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
      for i in xrange(len(new_sp)-1):
        gt_edge = self.G.edge(new_sp[i], new_sp[i+1])
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
    for i in xrange(len(path)-1):
      w = w + self.adj_mat[int(path[i]), int(path[i+1])]
    return w

  def in_edge_tup_form(self, path):
    edge_path = self.in_edge_form(path)
    edge_tup_path = [(self.G.vertex_index[e[0]], self.G.vertex_index[e[1]]) for e in edge_path]
    return edge_tup_path