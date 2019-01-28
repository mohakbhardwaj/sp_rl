#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import gym
import sp_rl
import networkx as nx
from agent import Agent
from graph import Graph
import time

class HeuristicAgent(Agent):
  def __init__(self, env, selector_str, ftr_params=None, lite_ftrs=False):
    super(HeuristicAgent, self).__init__(env)
    self.SELECTORS = {'select_expand': None,
                      'select_forward': self.select_forward,
                      'select_backward': self.select_backward,
                      'select_alternate': self.select_alternate,
                      'select_prior': self.select_prior,
                      'select_k_shortest': self.select_k_shortest,
                      'select_multiple': self.select_multiple,
                      'select_priorkshort': self.select_priorkshort,
                      'select_lookahead': self.select_lookahead,
                      'length_oracle':self.length_oracle}


    self.selector = self.SELECTORS[selector_str]
    _, self.graph_info = self.env.reset()
    self.ftr_params = ftr_params
    if 'pos' in self.graph_info:
      node_pos = self.graph_info['pos']
    else:
      node_pos = None
    self.G = Graph(self.graph_info['adj_mat'], self.graph_info['source_node'], self.graph_info['target_node'], ftr_params=ftr_params, pos=node_pos, edge_priors=self.graph_info['edge_priors'], lite_ftrs=lite_ftrs)


  def train(self, num_episodes, render=False, step=False):
    return self.test(num_episodes, render, step)
    

  def test(self, num_episodes, render=True, step=False):
    start_t = time.time()
    test_rewards = {}
    test_avg_rewards = {}
    avg_time_taken = 0.0
    j = 0
    for i in range(num_episodes):
      obs, _ = self.env.reset()
      self.G.reset()
      path = self.get_path(self.G)
      if render:
        self.render_env(obs, path)
      run_base = False 
      ep_reward = 0
      done = False
      while not done:
        path = self.get_path(self.G)
        feas_actions = self.filter_path(path, obs)
        act_id = self.selector(feas_actions, j+1, self.G)
        act_e = self.env.edge_from_action(act_id)
        # print feas_actions, act_e
        if render:
          self.render_env(obs, feas_actions, act_id)
        ftrs = self.G.get_features([self.env.edge_from_action(a) for a in feas_actions], j)
        # print ftrs
        # print('feasible actions = {}, chosen edge = {}, edge_features = {}'.format(feas_actions, [act_id, act_e], ftrs))
        if step: raw_input('Press enter to execute action')
        obs, reward, done, info = self.env.step(act_id)
        # print path, feas_actions, obs[act_id]
        # print ('obs = {}, reward = {}, done = {}'.format(obs, reward, done))
         
        self.G.update_edge(act_e, obs[act_id])#Update the edge weight according to the result
        ep_reward += reward
        j += 1
        avg_time_taken = time.time() - start_t

      #Render environment one last time
      if render:
        self.render_env(obs, path)
        raw_input('Press Enter')
      
      test_rewards[self.env.world_num] = ep_reward
      test_avg_rewards[self.env.world_num] = np.mean(test_rewards.values())
      if step: raw_input('Episode over, press enter for next episode')
    
    avg_time_taken = avg_time_taken/num_episodes*1.0
    
    return test_rewards, test_avg_rewards, avg_time_taken


  
  def render_env(self, obs, path, act_id=-1):
    edge_widths={}
    edge_colors={}
    for i in path:
      edge_widths[i] = 4.0
      edge_colors[i] = str(0.4)
      if i == act_id:
        edge_widths[i] = 7.0
        edge_colors[i] = str(0.0)
    self.env.render(edge_widths=edge_widths, edge_colors=edge_colors)
  
  def select_forward(self, feas_actions, iter, G):
    return feas_actions[0]

  def select_backward(self, feas_actions, iter, G):
    return feas_actions[-1]

  def select_alternate(self, feas_actions, iter, G):
    if iter%2 == 0:
      return feas_actions[-1]
    else:
      return feas_actions[0]

  def select_prior(self, feas_actions, iter, G):
    "Choose the edge most likely to be in collision"
    edges = list(map(self.env.edge_from_action, feas_actions))
    priors = G.get_priors(edges)
    idx_prior = np.argmax(priors)
    return feas_actions[idx_prior]

  def select_k_shortest(self, feas_actions, iter, G):
    "Choose the edge most likely to be in collision"
    edges = list(map(self.env.edge_from_action, feas_actions))
    kshort = G.get_k_short_num(edges)
    return feas_actions[np.argmax(kshort)]

  def select_multiple(self, feas_actions, iter, G):
    """Choose the best edge according to all heuristics"""
    edges = list(map(self.env.edge_from_action, feas_actions))
    ftrs = G.get_features(edges, iter)
    h = np.min(ftrs, axis=1)
    return feas_actions[np.argmax(h)]

  def select_priorkshort(self, feas_actions, iter, G):
    edges = list(map(self.env.edge_from_action, feas_actions))
    priors = G.get_priors(edges)
    kshort = [num*1.0/self.ftr_params['k']*1. for num in G.get_k_short_num(edges)]
    h = map(lambda x: x[0]*x[1], zip(kshort,priors))
    return feas_actions[np.argmax(h)]
  
  def select_lookahead(self, feas_actions, iter, G):
    k_sp_nodes, k_sp, k_sp_len = G.curr_k_shortest()
    scores = [0]*len(feas_actions)
    # print k_sp[0], G.curr_sp, len(k_sp)
    for j, action in enumerate(feas_actions):
      edge = self.env.edge_from_action(action)
      # print "edge", edge, self.env.G[edge[0]][edge[1]]['status']
      if self.env.G[edge[0]][edge[1]]['status'] == 0:
        #do something
        for sp in k_sp:
          if (edge[0],edge[1]) in sp or (edge[1],edge[0]) in sp:
            scores[j] += 1
      else:
        continue
    # print scores
    return feas_actions[np.argmax(scores)]

  def length_oracle(self, feas_actions, iter, G, horizon=2):
    curr_sp = G.curr_shortest_path
    curr_sp_len = G.curr_shortest_path_len
    scores = np.array([0.0]*len(feas_actions))
    for (j, action) in enumerate(feas_actions):
      gt_edge = self.env.gt_edge_from_action(action)
      edge = (gt_edge.source(), gt_edge.target())
      if self.env.G.edge_properties['status'][gt_edge] == 0:
        G.update_edge(edge, 0)
        new_sp = G.curr_shortest_path
        new_sp_len = G.curr_shortest_path_len
        scores[j] = scores[j] + (new_sp_len-curr_sp_len)
        G.update_edge(edge, -1)
    action = feas_actions[np.argmax(scores)]
    return action

  def get_score(self, feas_actions, iter, G, horizon=2):
    curr_sp = G.curr_shortest_path
    print(G.in_edge_form(curr_sp))
    curr_sp_len = G.curr_shortest_path_len
    scores = [0.0]*len(feas_actions)
    for (j, action) in enumerate(feas_actions):
      # edge = self.env.edge_from_action(action)
      gt_edge = env.gt_edge_from_action(action)
      if env.G.edge_properties['status'][gt_edge] == 0:
        G.update_edge(edge, 0)
        new_sp = G.curr_shortest_path
        new_sp_len = G.curr_shortest_path_len
        scores[j] = scores[j] + (new_sp_len-curr_sp_len)
        G.update_edge(edge, -1)
    # print('length oracle score'), scores
    return scores