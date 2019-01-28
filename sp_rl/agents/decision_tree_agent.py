#!/usr/bin/env python
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import gym
import sp_rl
import networkx as nx
from agent import Agent

class DecisionTreeAgent(Agent):
  def __init__(self, env):
    self.env = env
    obs, self.graph_info = self.env.reset()
    sp = self.get_path(obs)
    print sp
    self.build_trees(sp, obs)
    # print self.env.edge_from_action(1)
    # print self.env.edge_from_action(0)
  # def train(self, num_episodes, render=False):
  #   rewards = []
  #   avg_rewards = []
  #   for i in xrange(num_episdes):
  #     G = nx.from_numpy_matrix(graph_info['adj_mat'])
  #     path = self.get_path(G, graph_info, obs)
  #     if render:
  #       self.render_env(obs, path)
  #     ep_reward=0
  #     done = False
  #     while not done:
  #       pass
  #   return rewards, avg_rewards
    

  # def test(self, num_episodes, render=True):
  #   test_rewards = []
  #   test_avg_rewards = []
  #   return test_rewards, test_avg_rewards
  
  def build_trees(self, path, obs):
    root_list = [{'idx': i} for i in path]
    self.split(root_list, 0, obs, self.env.action_space.n)
    # self.print_trees(root_list)
    print self.print_trees([root_list[1]])
    # self.print_trees([root_list[1]])

  def split(self, node_list, depth, obs, max_depth):
    for node in node_list:
      #Branch on 0
      obs_temp_0 = deepcopy(obs)
      obs_temp_0[node['idx']] = 0
      if self.is_terminal_0(obs_temp_0): #or depth >= max_depth:
        node[0] = "invalid"
        continue
      node[0] = self.get_path_nodes(obs_temp_0)
      self.split(node[0], depth+1, obs_temp_0, max_depth)
    
    #Branch on 1
    for node in node_list:
      obs_temp_1 = deepcopy(obs)
      obs_temp_1[node['idx']] = 1
      if self.is_terminal_1(obs_temp_1):
        node[1] = "found"
        continue
      node[1] = self.get_path_nodes(obs_temp_1)
      self.split(node[1], depth+1, obs_temp_1, max_depth)

  
  def is_terminal_0(self, obs):
    #Node is terminal if feasible path to goal is found or no collision free path left
    path = self.get_path(obs)
    invalid = False
    if len(path) == 0: return True
    for i in path:
      if obs[i] == 0:
        invalid = True
        break 
    return invalid
  
  def is_terminal_1(self, obs):
    path = self.get_path(obs)
    if len(path) == 0: return True
    n_e = len(path)
    return sum([obs[i] for i in path]) == n_e

  
  def get_path(self, obs):
    G = nx.from_numpy_matrix(self.graph_info['adj_mat'])
    for i, o in enumerate(obs):
      if o == 0:
        edge = self.env.edge_from_action(i)
        G[edge[0]][edge[1]]['weight'] = np.inf
    sp = []
    if nx.shortest_path_length(G, self.graph_info['source_node'], self.graph_info['target_node'], 'weight') < np.inf:
      sp = nx.shortest_path(G, self.graph_info['source_node'], self.graph_info['target_node'], 'weight')
      sp = self.env.to_edge_path(sp) 
    return sp

  def get_path_nodes(self, obs):
    path = self.get_path(obs)
    path_nodes = []
    for edge in path:
      if obs[edge] == -1:
        path_nodes.append({'idx': edge})
    return path_nodes
  
  def print_trees(self, node_list):
    for i, node in enumerate(node_list):
      if 0 in node:
        if isinstance(node[0], list):
          print('Edge id: {}, 0 children {}'.format(node['idx'], [n['idx'] for n in node[0]]))
          self.print_trees(node[0])
        else:
          print('Edge id: {}, 0 children {}'.format(node['idx'], node[0]))
      if 1 in node:
        if isinstance(node[1], list):
          print('Edge id: {}, 1 children {}'.format(node['idx'], [n['idx'] for n in node[1]]))
          self.print_trees(node[1])
        else:
          print('Edge id: {}, 1 children {}'.format(node['idx'], node[1]))
  
  # def predict(self, node, )
