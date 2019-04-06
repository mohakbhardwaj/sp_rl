#!/usr/bin/env python
"""Weighted graph class that agent uses to represent the problem and calculate features"""

from graph_tool.all import *
import numpy as np
from itertools import islice
import time
from copy import deepcopy

class GraphWrapper(object):
  def __init__(self, adj_mat, source, target, ftr_params=None, pos=None, train_edge_statuses=None, lite_ftrs=True):
    # self.dist_fns = {'euc_dist': self.euc_dist}
    self.adj_mat = adj_mat
    self.pos = pos
    self.source = source
    self.target = target
    self.train_edge_statuses = train_edge_statuses
    print('Here', self.train_edge_statuses.shape)
    self.G = self.to_graph_tool(self.adj_mat) #Graph(directed=False)
    self.nedges = self.G.num_edges()
    estat = self.G.new_edge_property("int")
    self.G.edge_properties['status'] = estat
    # if pos is not None:
    #   vert_pos = self.G.new_vertex_property("vector<double>")
    #   self.G.vp['pos'] = vert_pos
    #   for v in self.G.vertices():
    #     vert_pos[v] = pos[v]

    # if edge_priors is not None:
    #   eprior = self.G.new_edge_property("double")
    #   self.G.edge_properties['p_free'] = eprior
    if self.train_edge_statuses is not None:
      self.edge_prior_vec = self.prior(self.train_edge_statuses)    
    for edge in self.G.edges():
      self.G.edge_properties['status'][edge] = -1
      # if edge_priors is not None:
      #   eprior[edge] = edge_priors[(edge.source(), edge.target())]
    
    self.curr_sp = self.shortest_path(self.source, self.target, 'weight', 'euc_dist')
    self.curr_sp_len = self.path_length(self.in_edge_form(self.curr_sp))
    self.num_invalid_checked = 0.0
    self.num_valid_checked = 0.0
    self.total_checked = 0.0
    self.lite_ftrs = lite_ftrs
    if self.lite_ftrs: self.num_features = 4
    else:
      self.num_features = 10 
      self.ftr_params = ftr_params
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
    self.curr_sp = self.shortest_path(self.source, self.target, 'weight', 'euc_dist')
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


  def shortest_path(self, source, target, attr='weight', dist_fn = 'euc_dist'):
    sp, _ = shortest_path(self.G, self.G.vertex(source), self.G.vertex(target), self.G.edge_properties['weight'])  
    return sp
  
  def prior(self, train_edge_stats):
    return 1.0 - np.mean(train_edge_stats, axis=0)
    


  def update_edge(self, edge, status, prev_stat=-1):
    gt_edge = self.G.edge(edge[0], edge[1])
    self.G.edge_properties['status'][gt_edge] = status #update edge status
    if status == 0: 
      self.G.edge_properties['weight'][gt_edge] = np.inf #if invalid set weight to infinity
      if prev_stat == -1:
        self.num_invalid_checked += 1.0
        self.total_checked += 1.0 
      self.curr_sp = self.shortest_path(self.source, self.target, 'weight', 'euc_dist') #recalculate shortest path if edge invalid
      self.curr_sp_len = self.path_length(self.in_edge_form(self.curr_sp))
    
    if status == -1: 
      self.G.edge_properties['weight'][gt_edge] = self.adj_mat[int(edge[0]), int(edge[1])]
      if prev_stat == 0:
        self.num_invalid_checked -= 1.0
        self.total_checked -= 1.0
      if prev_stat == 1:
        self.num_valid_checked -=1.0
        self.total_checked -=1.0
      self.curr_sp = self.shortest_path(self.source, self.target, 'weight', 'euc_dist') #recalculate shortest path if edge invalid
      self.curr_sp_len = self.path_length(self.in_edge_form(self.curr_sp))

      #recalcuate features if required
      # if not self.lite_ftrs:
      #   ksp_nodes_temp = []
      #   ksp_temp = []
      #   ksp_len_temp = []
        
        
      #   for i, sp in enumerate(self.k_sp):
      #     if (edge[0],edge[1]) not in sp and (edge[1], edge[0]) not in sp:
      #       ksp_nodes_temp.append(self.k_sp_nodes[i])
      #       ksp_temp.append(self.k_sp[i])
      #       ksp_len_temp.append(self.k_sp_len[i])

      #   if len(ksp_nodes_temp) == 0: #all removed calculate new
      #     print('Recomputing {} shortest paths'.format(self.ftr_params['k']))
      #     self.k_sp_nodes, self.k_sp, self.k_sp_len = self.k_shortest_paths(self.ftr_params['k'], attr='weight')
      #   else:
      #     self.k_sp_nodes = deepcopy(ksp_nodes_temp)
      #     self.k_sp = deepcopy(ksp_temp)
      #     self.k_sp_len = deepcopy(ksp_len_temp)

      #   nx.set_edge_attributes(self.G, 0.0, 'k_short_num')
      #   nx.set_edge_attributes(self.G, -1.0, 'k_short_len')

      #   for sp, l in zip(self.k_sp, self.k_sp_len):
      #     for e in sp:
      #       self.G[e[0]][e[1]]['k_short_num'] = self.G[e[0]][e[1]]['k_short_num'] + 1.0
      #       if self.G[e[0]][e[1]]['k_short_len'] == -1.0:
      #         self.G[e[0]][e[1]]['k_short_len'] = l
      #       else: self.G[e[0]][e[1]]['k_short_len'] = self.G[e[0]][e[1]]['k_short_len'] + l
      #       self.G[e[0]][e[1]]['k_short_eval'] = 0.0           

    elif status == 1:
      self.num_valid_checked += 1.0
      self.total_checked += 1.0

  def edge_features(self, edge, idx=-1, num_path_edges=np.inf, itr=0):
    prior = self.edge_prior_vec[edge]
    forward_score = 1.0
    # backward_score = 1.0
    if num_path_edges > 1:
      forward_score = 1.0 - (idx*1.0/(num_path_edges-1.0))
      # backward_score = idx*1./(num_path_edges-1.0)
    forward_oh = 0.0
    backward_oh = 0.0
    alt_oh = 0.0

    if idx == 0: forward_oh = 1.0
    if idx == num_path_edges-1: backward_oh = 1.0
    if itr%2 == 0: alt_oh = backward_oh
    else: alt_oh = forward_oh

    delta_len, delta_progress = self.get_edge_util(edge)

    if self.lite_ftrs:
      return  np.array([p_free, 
              forward_score,
              delta_len,
              delta_progress])
    
    # k_short_num = attr_dict['k_short_num']
    # k_short_len = attr_dict['k_short_len']
    # if len(self.k_sp) > 0 and attr_dict['k_short_num'] > 0:
    #   k_short_num = attr_dict['k_short_num']/len(self.k_sp)
    #   k_short_len = (attr_dict['k_short_len']/attr_dict['k_short_num']) / (sum(self.k_sp_len)/len(self.k_sp))

    valid_checked = 0.0
    invalid_checked = 0.0
    
    if self.total_checked > 0:
      valid_checked = self.num_valid_checked/self.total_checked
      invalid_checked = self.num_invalid_checked/self.total_checked
    return np.array([prior, 
                     forward_score,
                     delta_len,
                     delta_progress,
                     forward_oh,
                     backward_oh,
                     alt_oh,
                     valid_checked,
                     invalid_checked])

  # def curr_k_shortest(self):
    # return self.k_sp_nodes, self.k_sp, self.k_sp_len


  def get_features_old(self, edges, itr):
    features = np.array([self.edge_features(edge, i, len(edges), itr) for i, edge in enumerate(edges)])
    #Normalize the delta_len feature
    if self.num_features >=3:
      if np.max(features[:,2]) > 0:
        features[:,2] = features[:,2]/np.max(features[:,2])
    if not self.lite_ftrs: #When not using lite_ftrs, we use one-hot vectors
      prior_oh = np.zeros(shape=(features.shape[0], 1))
      prior_oh[np.argmax(features[:,0])] = 1.0
      features_final = np.concatenate((features, prior_oh), axis=1)
    else:
      features_final = features
    return features_final

  def get_features(self, edge_ids, obs, itr):
    priors = self.edge_prior_vec[idxs]
    forward_scores = 1.0 - np.arange(edge_ids.size())/edge_ids.size()
    feature_vecs = np.concatenate(priors, forward_scores)
    return feature_vecs

  def get_priors(self, idxs):
    priors = self.edge_prior_vec[idxs]
    return priors

  def get_posterior(self, idxs, obs):
    """Posterior over edges using current uncovered world to calculate probability of a training world
       to be the real one.
       train_edges: N X E binary matrix where N is number of worlds and E is number of edges.
       obs: 1XE with 0 for known invalid, 1 for known valid and -1 for unknown.
    """
    obs_idxs = np.where(obs >= 0)[0] #get idxs of edges that have been checked
    obs_checked = obs[obs_idxs]
    train_edges_checked = self.train_edge_statuses[:, obs_idxs]
    #Calculate deviation between training dataset and current observation  
    deviation = np.abs(train_edges_checked - obs_checked) 
    deviation = np.sum(deviation, axis=1)
    #Calculate probability of each world being true using softmax 
    scores = -1.0*deviation
    exp_scores = np.exp(scores)
    probs = exp_scores/np.sum(exp_scores)
    #Calculate the posterior (expected value of status of an edge)
    posterior = self.train_edge_statuses * probs.reshape(probs.shape[0],1)
    posterior = np.sum(posterior , axis=0)
    post = 1.0 - posterior[idxs]
    # print('probs', probs[1])  
    # print('posterior', post)
    # print('priors', self.get_priors(idxs))
    return post    

  def get_utils(self, edges):
    delta_lens = []
    delta_progs = []
    for edge in edges:
      dl, dp = self.get_edge_util(edge)
      delta_lens.append(dl)
      delta_progs.append(dp)
    return delta_lens, delta_progs




  # def get_k_short_num(self, edges):
  #   return list(map(lambda e: self.G[e[0]][e[1]]['k_short_num'], edges))
  
  # def get_k_short_len(self, edges):
    # return list(map(lambda e: self.G[e[0]][e[1]]['k_short_len'], edges))

  def get_edge_util(self, query_edge):
    curr_sp = self.curr_sp
    curr_sp_len = self.curr_sp_len
    self.update_edge(query_edge, 0, -1)
    new_sp = self.curr_sp

    if len(new_sp) > 0:
      new_sp_len = self.curr_sp_len
      delta_len = new_sp_len - curr_sp_len
      delta_progress = 0.0
      for e in self.in_edge_form(new_sp):
        gt_edge = self.G.edge(e[0], e[1])
        if self.G.edge_properties['status'][gt_edge] == -1:
          # print e
          delta_progress += 1.0
      delta_progress = delta_progress/(len(new_sp)-1)
    else: 
      delta_len = 0.0
      delta_progress = 0.0
    self.update_edge(query_edge, -1, 0)
    return delta_len, delta_progress

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

  # def euc_dist(self, node1, node2):
  #   pos1 = np.array(self.G.node[node1]['pos'])
  #   pos2 = np.array(self.G.node[node2]['pos'])
  #   return np.linalg.norm(pos1 - pos2)

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