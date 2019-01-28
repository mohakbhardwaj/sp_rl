#!/usr/bin/env python
import sys, os
# sys.path.insert(0, os.path.abspath('../..'))
import numpy as np
import networkx as nx
from copy import deepcopy
from itertools import islice
import time
# from sp_rl.agents import Graph


class SetCoverOracle(object):
  def __init__(self, env, G, horizon):#, path_file):#, k):
    # print self.path_to_id
    self.horizon = horizon
    self.G = G
    self.env = env
    self.true_sp = deepcopy(self.env.sp)
    self.true_sp_uneval = self.in_edge_form(deepcopy(self.env.sp))
    self.true_path_ctr = 0
    self.path_found = False
    self.sp_set = []
    self.elim_idxs = []

  def get_best_action(self):
    nx.set_edge_attributes(self.G.G, 0.0, 'score')
    nx.set_edge_attributes(self.G.G, [], 'elim_idxs')
    best_score = 0
    best_edge = None
    best_elim_idxs = []
   
    #I all paths in horizon have been eliminated, calculate next set of paths
    if len(self.sp_set) == 0:
      print "Regenerating paths"
      self.sp_set = list(islice(nx.shortest_simple_paths(self.G.G, self.env.source_node, self.env.target_node, 'weight'), self.horizon))
      self.elim_idxs = []

    #If optimal path has not been found, find the edge that eliminates most paths and remove those paths from path set
    if not self.path_found:
      for p, path in enumerate(self.sp_set):
        if p in self.elim_idxs: continue
        path_e = self.in_edge_form(path)
        # print p
        if path == self.true_sp:
          self.path_found = True
          print "Found path", path, self.true_sp
          break 
        else: #Check every edge on the path and increase score for invalid edges
          for edge in path_e:
            if self.env.G[edge[0]][edge[1]]['status'] == 0:
              self.G.G[edge[0]][edge[1]]['score'] = self.G.G[edge[0]][edge[1]]['score'] + 1
              self.G.G[edge[0]][edge[1]]['elim_idxs'] = self.G.G[edge[0]][edge[1]]['elim_idxs'] + [p]
            
              if self.G.G[edge[0]][edge[1]]['score'] >= best_score:
                best_score = self.G.G[edge[0]][edge[1]]['score']
                best_edge = edge
      
      if not self.path_found:          
        if best_edge == None:
          print self.sp_set[0], len(self.elim_idxs), self.true_sp
        best_elim_idxs = self.G.G[best_edge[0]][best_edge[1]]['elim_idxs']
        self.elim_idxs = self.elim_idxs + best_elim_idxs

      if len(self.elim_idxs) == self.horizon:
        self.sp_set = []  


    #If optimal path is at the top of the path set, expand edges along it.
    if self.path_found:
      best_edge = self.true_sp_uneval[self.true_path_ctr]
      best_score = 0
      self.true_path_ctr = self.true_path_ctr + 1
      # print best_edge, best_score
    return best_edge , best_score

  def rollout_oracle(self, obs, render=False, step=False):
    start_t = time.time()
    done = False
    value = 0
    best_action = None
    self.filter_out_edges(obs)
    self.sp_set = []
    self.elim_idxs = []
    while not done:
      edge, score = self.get_best_action()

      act_id = self.env.action_from_edge(edge)
      print act_id
      if render:
        self.render_env(self.env.to_edge_path(self.true_sp), act_id)
        if step:
          raw_input('Press enter to execute action')
      obs, reward, done, info = self.env.step(act_id)
      if not self.path_found:  
        self.G.G.remove_edge(edge[0], edge[1])
        # self.sp_set = islice(nx.shortest_simple_paths(self.G.G, self.env.source_node, self.env.target_node, 'weight'), self.horizon)
        
      value = value + reward
    time_taken = time.time() - start_t
    return value, time_taken
  
  def filter_out_edges(self, obs):
    for i in xrange(len(obs)):
      if obs[i] != -1:
        edge = self.env.edge_from_action(i)
        self.G.G.remove_edge(edge[0], edge[1])
  
  def in_edge_form(self, path):
    """Helper function that takes a path(set of nodes) as input
    and returns a path as a set of edges""" 
    return [(i,j) for i,j in zip(path[:-1], path[1:])]

  def render_env(self, path, act_id):
    edge_widths={}
    edge_colors={}
    for i in path:
      edge_widths[i] = 4.0
      edge_colors[i] = str(0.4)
    
    edge_widths[act_id] = 7.0
    edge_colors[act_id] = str(0.0)
    self.env.render(edge_widths=edge_widths, edge_colors=edge_colors)