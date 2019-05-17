#!/usr/bin/env python
import operator
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import sp_rl
import networkx as nx
from agent import Agent
from copy import deepcopy
from sp_rl.learning import LinearNet

class CEMAgent(Agent):
  def __init__(self, train_env, valid_env, model, G, batch_size, elite_frac, init_std):
    super(CEMAgent, self).__init__(train_env)
    self.train_env = train_env
    self.valid_env = valid_env
    self.model = model
    self.G = G
    self.batch_size = batch_size
    self.elite_frac = elite_frac
    self.init_std = init_std
    self.n_elite = int(np.round(self.batch_size*self.elite_frac))


  def train(self, num_iters, num_episodes_per_iter, num_valid_episodes, quad_ftrs=False):
    validation_rewards   = np.zeros(shape=(num_iters, num_valid_episodes)) 
    weights_per_iter = {}
    th_mean = self.model.get_weights().detach().numpy()
    th_std = np.ones_like(th_mean) * self.init_std  
    episodes_per_batch = int(num_episodes_per_iter*1.0/self.batch_size)
    # self.train_env.reset(roll_back=True)
    
    if num_valid_episodes == 0:
      best_valid_reward = -np.inf
    else: 
      print('Evaluating initial model on validation')
      valid_rewards_dict = self.test(self.valid_env, self.model, num_valid_episodes, render=False, step=False, quad_ftrs=quad_ftrs)
      valid_rewards = [it[1] for it in sorted(valid_rewards_dict.items(), key=operator.itemgetter(0))]
      best_valid_reward = np.median(valid_rewards)
      print('Best valid reward', best_valid_reward)

    for i in range(num_iters):
      self.train_env.reset(roll_back=True)
      print('Iteration = {}, Model weights'.format(i))
      self.model.print_parameters()
      # print th_mean, th_std
      ths = np.array([th_mean + dth for dth in th_std*np.random.randn(self.batch_size, th_mean.size)])
      ys = np.array([self.noisy_evaluation(self.train_env, th, episodes_per_batch, quad_ftrs) for th in ths])
      # print ys
      elite_inds = ys.argsort()[::-1][:self.n_elite]
      elite_ths  = ths[elite_inds]
      th_mean    = elite_ths.mean(axis=0)
      th_std     = elite_ths.std(axis=0)
      weights_per_iter[i] = th_mean.tolist()
      

      if num_valid_episodes == 0:
        self.model.set_weights(torch.tensor(th_mean))
        self.model.print_parameters()
        median_reward = 0
      else:
        model_curr = deepcopy(self.model)
        model_curr.set_weights(torch.tensor(th_mean))
        valid_rewards_dict = self.test(self.valid_env, model_curr, num_valid_episodes, render=False, step=False, quad_ftrs=quad_ftrs)
        valid_rewards = [it[1] for it in sorted(valid_rewards_dict.items(), key=operator.itemgetter(0))]
        median_reward = np.median(valid_rewards)
        print ('Validation reward', median_reward)
        if median_reward >= best_valid_reward:
          print('Best policy yet, saving')
          self.model = deepcopy(model_curr)
          self.model.print_parameters()
          best_valid_reward = median_reward
        else: print('Keeping old policy')
        validation_rewards[i,:]   = valid_rewards
    return validation_rewards, weights_per_iter



  def test(self, env, model, num_episodes, render=False, step=False, quad_ftrs=False):
    test_rewards = {}
    _, _ = env.reset(roll_back=True)
    model.eval() #Put the model in eval mode to turn off dropout etc.
    model.cpu() #Copy over the model to cpu if on cuda 
    with torch.no_grad(): #Turn off autograd
      for i in xrange(num_episodes):
        j = 0
        ep_reward = 0
        obs, _ = env.reset()
        self.G.reset()
        done = False

        if render:
          path = self.get_path(self.G)
          feas_actions = self.filter_path(path, obs)
          ftrs = torch.tensor(self.G.get_features(feas_actions, obs, j, quad_ftrs))
          scores = model(ftrs).detach().numpy()
          self.render_env(env, path, scores)

        while not done:
          path = self.get_path(self.G)
          feas_actions = self.filter_path(path, obs)
          ftrs = torch.tensor(self.G.get_features(feas_actions, obs, j, quad_ftrs))
          scores = model(ftrs).detach().numpy()
          if render:
            self.render_env(env, feas_actions, scores)
          idx = np.random.choice(np.flatnonzero(scores==scores.max())) #random tie breaking
          act_id = feas_actions[idx]
          if step: raw_input('Press enter to execute action')
          obs, reward, done, info = env.step(act_id)
          self.G.update_edge(act_id, obs[act_id])#Update the edge weight according to the results                  
          ep_reward += reward
          j += 1
        # print('Final path = {}'.format([env.edge_from_action(a) for a in path]))

        if render:
          fr = torch.tensor(self.G.get_features(path, obs, j, quad_ftrs))
          sc = policy.predict(fr)
          self.render_env(env, path, sc.detach().numpy())
          raw_input('Press Enter')
        if step: raw_input('Episode over, press enter for next episode')
        test_rewards[env.world_num] = ep_reward
    return test_rewards  



  def noisy_evaluation(self, env, th, num_episodes, quad_ftrs):
    rewards = np.zeros(num_episodes)
    with torch.no_grad(): #Turn off autograd
      model = deepcopy(self.model)
      model.set_weights(torch.tensor(th))
      #print('Evaluating parameters', th)
      #model.print_parameters()
      for i in xrange(num_episodes):
        j = 0
        ep_reward = 0
        obs, _ = env.reset()
        self.G.reset()
        done = False
        while not done:
          path = self.get_path(self.G)
          feas_actions = self.filter_path(path, obs)
          ftrs = torch.tensor(self.G.get_features(feas_actions, obs, j, quad_ftrs))
          scores = model(ftrs).detach().numpy()
          idx = np.random.choice(np.flatnonzero(scores==scores.max())) #random tie breaking
          act_id = feas_actions[idx]
          obs, reward, done, info = env.step(act_id)
          self.G.update_edge(act_id, obs[act_id])#Update the edge weight according to the results                  
          ep_reward += reward
          j += 1
        # print('Final path = {}'.format([env.edge_from_action(a) for a in path]))
        rewards[i] = ep_reward
    med_reward = np.median(rewards)
    #print('median reards = {}'.format(med_reward))
    return med_reward




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


