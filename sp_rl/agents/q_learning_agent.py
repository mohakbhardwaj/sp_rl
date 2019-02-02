#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import sp_rl
from agent import Agent
from sp_rl.learning import LinearNet, ExperienceBuffer


class QLearningAgent(Agent):
  def __init__(self, env, qfun, G, eps0, epsf, min_data):
    super(QLearningAgent, self).__init__(env)
    self.env = env
    self.eps0 = eps0
    self.epsf = epsf
    print self.epsf
    # _, self.graph_info = self.env.reset()
    self.G = G
    self.q_fun = qfun
    # self.num_weights = self.G.num_features
    # w_init = np.random.rand(self.num_weights, 1)
    # self.w_init = w_init#np.ones(shape=(self.num_weights, 1))
    # b_init = 1
    # self.q_fun = LinearQFunction(w_init, b_init, alpha)


  def train(self, num_episodes, num_exp_episodes, render=False, step=True):
    train_rewards=[]
    train_avg_rewards = []
    train_loss = []
    train_avg_loss = []
    # print self.eps0, self.epsf, num_exp_episodes
    eps_sc = self.eps_schedule(num_exp_episodes, self.eps0, self.epsf)
    self.env.reset(roll_back=True)
    D = ExperienceBuffer() 
    
    for i in xrange(num_episodes):
      j = 0
      if i%20 == 0:
        obs, _ = self.env.reset(roll_back=True)
      
      obs, _ = self.env.reset()
      self.G.reset()
      path = self.get_path(self.G)
      ftrs = torch.tensor(self.G.get_features([self.env.edge_from_action(a) for a in path], j))
      q_vals = self.q_fun.predict(ftrs)
      q_vals = q_vals.detach().numpy()
      if render:
        self.render_env(self.env, path, q_vals)
      
      ep_reward = 0
      ep_loss = 0

      if i < num_exp_episodes:
        eps = eps_sc[i]
      else:
        eps = self.epsf
      
      done = False
      while not done:
        path = self.get_path(self.G)
        feas_actions = self.filter_path(path, obs) #Only select from unevaluated edges in shortest path
        ftrs = torch.tensor(self.G.get_features([self.env.edge_from_action(a) for a in feas_actions], j))
        q_vals = self.q_fun.predict(ftrs) 
        q_vals = q_vals.detach().numpy() 
        #Epsilon greedy action selection
        if np.random.sample() > eps:
          idx = np.random.choice(np.flatnonzero(q_vals==q_vals.max())) #random tie breaking
          print "greedy action"
        else:
          idx = self.get_random_action(feas_actions, self.G) 
          print "random action"
        
        act_id = feas_actions[idx] 
        act_e = self.env.edge_from_action(act_id)
        act_q = q_vals[idx]
        act_ftrs = ftrs[idx]
        
        obs,reward, done, info = self.env.step(act_id)
        # print ('Q_vals = {}, feasible_actions = {}, chosen edge = {}, features = {}'.format(q_vals, feas_actions, act_e, ftrs))
        self.G.update_edge(act_e, obs[act_id]) #Update agent's internal graph
        
        #calculate best next action and push to buffer
        if not done:
          pathd = self.get_path(self.G)
          feas_actionsd = self.filter_path(pathd, obs)
          ftrsd = torch.tensor(self.G.get_features([self.env.edge_from_action(a) for a in feas_actionsd], j))
          q_valsd = self.q_fun.predict(ftrs)
          q_valsd = q_valsd.detach().numpy()
          act_ftrsd = q_valsd[np.argmax(q_valsd)]
          # target = self.get_q_target(reward, pathd, obs, j+1)
          D.push(act_ftrs, reward, ftrsd)
        else:
          D.push(act_ftrs, reward, act_ftrs.unsqueeze(0))

        if len(D) >= self.q_fun.batch_size:
          train_loss = self.q_fun.train(D)
        ep_reward += reward
        j += 1
      print('Curr episode = {}'.format(i+1))
      print('Episode reward = %d'%(ep_reward))
      train_rewards.append(ep_reward)

      # train_avg_loss.append(sum(train_loss)*1.0/(i+1.0))
      # train_rewards_dict[i] = ep_reward
      # train_avg_rewards.append(sum(train_rewards)*1.0/(i+1.0))
      
    return train_rewards#, train_avg_rewards #train_loss, train_avg_loss



  def test(self, num_episodes, render=True, step=False):
    test_rewards = []
    test_avg_rewards = []
    test_loss = []
    test_avg_loss = []
    for i in range(num_episodes):
      j = 0
      obs, _ = self.env.reset()
      self.G.reset()
      path = self.get_path(self.G)
      ftrs = self.G.get_features([self.env.edge_from_action(a) for a in path], j)
      q_vals = self.q_fun.predict(np.array(ftrs))
      if render:
        self.render_env(path, q_vals)
      ep_reward = 0
      ep_loss = 0
      done = False
      while not done:
        print('Curr episode = {}'.format(i+1))
        path = self.get_path(self.G)
        feas_actions = self.filter_path(path, obs)
        ftrs = self.G.get_features([self.env.edge_from_action(a) for a in feas_actions], j)
        q_vals = self.q_fun.predict(np.array(ftrs))
        if render:
          self.render_env(feas_actions, q_vals)
        #select greedy action
        idx = np.random.choice(np.flatnonzero(q_vals==q_vals.max())) #random tie breaking
        act_id = feas_actions[idx]

        act_e = self.env.edge_from_action(act_id)
        act_q = q_vals[idx]
        # print ('Q_vals = {}, feasible_actions = {}, chosen edge = {}, features = {}'.format(q_vals, feas_actions, act_e, ftrs))
        if step: raw_input('Press enter to execute action')
        obs, reward, done, info = self.env.step(act_id)
        # print ('obs = {}, reward = {}, done = {}'.format(obs, reward, done))
         
        self.G.update_edge(act_e, obs[act_id])#Update the edge weight according to the result
        
        #Get q error 
        pathd = self.get_path(self.G)
        target = self.get_q_target(reward, pathd, obs, j+1)
        loss = (target - act_q) ** 2
        
        ep_reward += reward
        ep_loss += loss
        j += 1

      print('Final path = {}'.format([self.env.edge_from_action(a) for a in path]))
      #Render environment one last time
      if render:
        fr = self.G.get_features([self.env.edge_from_action(a) for a in path], j)
        qr = self.q_fun.predict(np.array(fr))
        self.render_env(path, qr)
        raw_input('Press Enter')

      ep_loss = ep_loss/(j*1.0)
      test_loss.append(ep_loss)
      test_avg_loss.append(sum(test_loss)/(i+1.0)) 
      test_rewards.append(ep_reward)
      test_avg_rewards.append(sum(test_rewards)*1.0/(i+1.0))
      if step: raw_input('Episode over, press enter for next episode')\

    return test_rewards, test_avg_rewards, test_loss, test_avg_loss
  
  def render_env(self, env, path, scores):
    # scores=scores.numpy()
    edge_widths={}
    edge_colors={}
    for k,i in enumerate(path):
      edge_widths[i] = 5.0
      color_val = 1.0 - (scores[k][0] - np.min(scores))/(np.max(scores) - np.min(scores))
      color_val = max(min(color_val, 0.65), 0.1)
      edge_colors[i] = str(color_val)
      # print color_val
    env.render(edge_widths=edge_widths, edge_colors=edge_colors)
  
  
  def get_q_target(self, reward, pathd, obs, itr):
    pathd_f = self.filter_path(pathd, obs)
    if len(pathd_f) > 0: #if optimal path has not been found
      ftrsd = self.G.get_features([self.env.edge_from_action(a) for a in pathd_f], itr)
      q_valsd = self.q_fun.predict(np.array(ftrsd))
      ad = np.argmax(q_valsd)
      target = reward + self.gamma * np.max(q_valsd)
    else:
      target = 0.0
    return target



  def eps_schedule(self, decay_episodes, eps0, epsf):
    sc = np.linspace(epsf, eps0, decay_episodes)
    return sc[::-1]
  

  def select_prior(self, feas_actions, G):
    "Choose the edge most likely to be in collision"
    edges = list(map(self.env.edge_from_action, feas_actions))
    priors = G.get_priors(edges)
    return np.argmax(priors)

  def get_random_action(self, feas_actions, G):
    idx = np.random.randint(len(feas_actions)) #select purely random action
    return idx


