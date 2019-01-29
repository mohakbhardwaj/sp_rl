#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import gym
import sp_rl
import networkx as nx
from agent import Agent
from graph_wrapper import GraphWrapper

class QTableAgent(Agent):
  def __init__(self, env, eps0, gamma, alpha):
    super(QTableAgent, self).__init__(env)
    self.eps0 = eps0
    self.gamma = gamma
    self.alpha = alpha
    self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n), dtype=np.float32) - self.env.action_space.n #Initialize Q table
    _, self.graph_info = self.env.reset()
    self.G = Graph(self.graph_info['adj_mat'], self.graph_info['source_node'], self.graph_info['target_node'], ftr_params=dict(k=75), pos=self.graph_info['pos'], edge_priors=self.graph_info['edge_priors'])

    
  def train(self, num_episodes, num_exp_episodes, render=False):
    rewards = []
    avg_rewards = []
    for i in range(num_episodes):
      obs, graph_info = self.env.reset()
      #Create a graph here
      self.G.reset()# = Graph(graph_info['adj_mat'], graph_info['source_node'], graph_info['target_node'], ftr_params=dict(k=50), pos=graph_info['pos'], edge_priors=graph_info['edge_priors'])
      path = self.get_path(self.G)
      if render:
        self.render_env(obs, path)
      ep_reward = 0
      eps = self.eps0
      if i >= num_exp_episodes:
        eps = self.eps0/(i+1.0) #Decay epsilon after the exploration phase is over
      done = False
      while not done:
        print('Curr episode = {}'.format(i+1))
        s = self.env.obs_to_id(obs)
        path = self.get_path(self.G)
        feas_actions = self.filter_path(path, obs)
        if np.random.sample() > eps:
          Q_set = self.Q[s,feas_actions] #Only select from unevaluated edges in shortest path
          idx = np.random.choice(np.flatnonzero(Q_set==Q_set.max())) #random tie breaking
          act_id = feas_actions[idx]
          print "greedy action"
        else:
          act_id = np.random.choice(feas_actions) #Choose a random unevaluated edge from the path
          print "random action"
        act_e = self.env.edge_from_action(act_id)
        ftrs = self.G.get_features([act_e])
        obs, reward, done, info = self.env.step(act_id)
        print ('Q_vals = {}, feasible_actions = {}, chosen edge = {}, features = {}'.format(self.Q[s,:], feas_actions, act_e, ftrs))
        # print('obs = {}, reward = {}, done = {}'.format(obs, reward, done))
        #Update Q values here
        self.G.update_edge(act_e, obs[act_id])#Update the edge weight according to the result
 
        sd = self.env.obs_to_id(obs)
        path_d = self.get_path(self.G)
        feas_actions_d = self.filter_path(path_d, obs)
        Q_t = np.max(self.Q[sd,feas_actions_d]) if len(feas_actions_d) > 0 else 0
        self.Q[s][act_id] = self.Q[s][act_id] + self.alpha * (reward + self.gamma * Q_t - self.Q[s][act_id])
        ep_reward += reward
        if render:
          self.render_env(obs, path)

      rewards.append(ep_reward)
      avg_rewards.append(sum(rewards)*1.0/(i+1.0))
    return rewards, avg_rewards
    

  def test(self, num_episodes, render=True, step=False):
    test_rewards = []
    test_avg_rewards = []
    for i in range(num_episodes):
      obs, graph_info = self.env.reset()
      self.G.reset()# = Graph(graph_info['adj_mat'], graph_info['pos'], graph_info['edge_priors'])
      path = self.get_path(self.G)
      if render:
        self.render_env(obs, path)
      ep_reward = 0
      done = False
      while not done:
        print('Curr episode = {}'.format(i+1))
        #rendering environment
        s = self.env.obs_to_id(obs)
        path = self.get_path(self.G)
        if render:
          self.render_env(obs, path)
        
        feas_actions = self.filter_path(path, obs)
        Q_set = self.Q[s,feas_actions] #Only select from unevaluated edges in shortest path
        idx = np.random.choice(np.flatnonzero(Q_set==Q_set.max())) #random tie breaking
        act_id = feas_actions[idx]
        act_e = self.env.edge_from_action(act_id)
        print('Q_values = {}, feasible actions = {}, chosen edge = {}'.format(self.Q[s,:], feas_actions, [act_id, act_e]))
        if step: raw_input('Press enter to execute action')
        obs, reward, done, info = self.env.step(act_id)
        print ('obs = {}, reward = {}, done = {}'.format(obs, reward, done))
        if obs[act_id] == 0: self.G.update_edge(act_e, obs[act_id])# #Update the edge weight according to the result
        ep_reward += reward

      #Render environment one last time
      if render:
        self.render_env(obs, path)
      test_rewards.append(ep_reward)
      test_avg_rewards.append(sum(test_rewards)*1.0/(i+1.0))
      if step: raw_input('Episode over, press enter for next episode')
    return test_rewards, test_avg_rewards
  
  def render_env(self, obs, path):
    s = self.env.obs_to_id(obs)
    edge_widths={}
    edge_colors={}
    for i in path:
      edge_widths[i] = min(max(abs(13.0/(self.Q[s,i] + 0.001)), 4),13.0)#, 5.0)
      edge_colors[i] = str(0.0) if self.Q[s,i] == 0.0 else str(max(min(abs((self.Q[s,i]))/max(abs(self.Q[s,path])), 0.7), 0.1))
    self.env.render(edge_widths=edge_widths, edge_colors=edge_colors)
