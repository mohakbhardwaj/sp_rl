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
from sp_rl.learning import LinearNet, ExperienceBuffer, select_forward, select_backward, select_alternate, select_prior, select_posterior, select_ksp_centrality, select_posteriorksp, select_delta_len, select_delta_prog, select_posterior_delta_len, length_oracle, length_oracle_2, init_weights

class DaggerAgent(Agent):
  def __init__(self, train_env, valid_env, policy, beta0, gamma, expert_str, G, epochs):
    super(DaggerAgent, self).__init__(train_env)
    self.EXPERTS = {'select_expand': None,
                    'select_forward':   select_forward,
                    'select_backward':  select_backward,
                    'select_alternate': select_alternate,
                    'select_prior':     select_prior,
                    'select_posterior': select_posterior,
                    'select_ksp_centrality': select_ksp_centrality,
                    'select_posteriorksp':   select_posteriorksp,
                    'select_delta_len':  select_delta_len,
                    'select_delta_prog': select_delta_prog,
                    'select_posterior_delta_len': select_posterior_delta_len,
                    'length_oracle': length_oracle,
                    'length_oracle_2': length_oracle_2}
    
    self.train_env = train_env
    self.valid_env = valid_env
    self.expert = self.EXPERTS[expert_str]
    self.beta0 = beta0
    self.gamma = gamma
    self.G = G
    self.train_epochs = epochs
    self.policy = policy


  def train(self, num_iterations, num_eval_episodes, num_valid_episodes,
            heuristic = None, re_init=False, render=False, step=False, mixed_rollin=False, quad_ftrs=False):

    train_rewards = np.zeros(shape=(num_iterations, num_eval_episodes))
    train_accs = np.zeros(shape=(num_iterations, num_eval_episodes))
    train_loss = np.zeros(shape=num_iterations)
    validation_reward   = np.zeros(shape=(num_iterations, num_valid_episodes)) 
    validation_accuracy = np.zeros(shape=(num_iterations, num_valid_episodes))
    dataset_size        = np.zeros(shape=num_iterations) 
    weights_per_iter = {}
    features_per_iter = {}
    labels_per_iter = {}
    D = ExperienceBuffer() 
    policy_curr = deepcopy(self.policy)
    weights_per_iter[-1] = policy_curr.model.serializable_parameter_dict()
    policy_curr.model.print_parameters()

    best_valid_reward = -np.inf
    curr_median_reward = -np.inf
    for i in xrange(num_iterations):
      Xdata = []; Ydata = []
      if i == 0: beta = 1.0 #Rollout with expert initially
      else: beta = self.beta0/i*1.0      
      #Reset the environment to first world
      self.train_env.reset(roll_back=True)
      #Evaluate every policy for k episodes
      policy_curr.model.eval()
      policy_curr.model.cpu()
      with torch.no_grad():
        for k in xrange(num_eval_episodes):
          j = 0
          ep_reward = 0.0
          ep_acc = 0.0
          obs, _ = self.train_env.reset()
          self.G.reset()
          if render:
            path   = self.get_path(self.G)
            ftrs   = torch.tensor(self.G.get_features(feas_actions, obs, j, quad_ftrs))
            scores = policy_curr.predict(ftrs).detach().numpy()
            self.render_env(self.train_env, path, scores)
          done = False          
          while not done:
            path = self.get_path(self.G)
            feas_actions = self.filter_path(path, obs) #Only select from unevaluated edges in shortest path
            # print obs 
            ftrsnp = self.G.get_features(feas_actions, obs, j, quad_ftrs)
            ftrs   = torch.tensor(ftrsnp)
            scores = policy_curr.predict(ftrs).detach().numpy() 
            # scores = scores.detach().numpy()
            idx_l = np.random.choice(np.flatnonzero(scores==scores.max())) #random tie breaking
            idx_exp = self.expert(feas_actions, obs, j, self.G, self.train_env)
            #Aggregate the data
            if np.random.sample() <= self.gamma:
              print('Adding to dataset')
              targets = torch.zeros(len(feas_actions),1)
              targets[idx_exp] = 1.0
              D.push(ftrs, targets)            
              ftrsl  = ftrsnp.tolist()
              for id in xrange(len(feas_actions)):
                Xdata.append(ftrsl[id])
                if id == idx_exp: Ydata.append(1)
                else: Ydata.append(0)

            #Execute mixture
            if np.random.sample() > beta: idx = idx_l 
            else:
              idx = idx_exp
              if heuristic is not None: 
                idx_heuristic = self.EXPERTS[heuristic](feas_actions, obs, j, self.G, self.train_env) 
                if mixed_rollin: idx = np.random.choice([idx_exp, idx_heuristic])
                else: idx = idx_heuristic
                if idx == idx_heuristic: print('Heuristic action') 
            
            act_id = feas_actions[idx]
            print('Current iteration = {}, Iter episode = {}, Timestep = {}, Selected edge ={}, Expert edge = {},  beta = {}, Curr median reward = {}'.format(i+1, k+1, j, act_id, feas_actions[idx_exp], beta, curr_median_reward))
            obs, reward, done, info = self.train_env.step(act_id)
            self.G.update_edge(act_id, obs[act_id]) #Update agent's internal graph          
            j += 1
            ep_reward += reward
            if idx_l == idx_exp: ep_acc += 1.0
          train_rewards[i,k] = ep_reward
          train_accs[i,k] = ep_acc/(j*1.0)
        curr_median_reward = np.median(train_rewards[i])
      
      #re-initialize policy and train
      if re_init:
        print('Current policy parameters')
        policy_curr.model.print_parameters()
        policy_curr.model.apply(init_weights)
        print('Re-initialized policy parameters')
        policy_curr.model.print_parameters()
      
      
      curr_train_loss = policy_curr.train(D, self.train_epochs)
      policy_curr.model.print_parameters()
      dataset_size[i] = (len(D))
      weights_per_iter[i] = policy_curr.model.serializable_parameter_dict()
      features_per_iter[i] = Xdata
      labels_per_iter[i]   = Ydata
      train_loss[i] = curr_train_loss
      if num_valid_episodes == 0:
        self.policy = deepcopy(policy_curr)
        median_reward = 0
      else:
        print('Running validation')
        valid_rewards_dict, valid_acc_dict = self.test(self.valid_env, policy_curr, num_valid_episodes, quad_ftrs=quad_ftrs)
        valid_rewards = [it[1] for it in sorted(valid_rewards_dict.items(), key=operator.itemgetter(0))]
        valid_accs    = [it[1] for it in sorted(valid_acc_dict.items(), key=operator.itemgetter(0))]        
        median_reward = np.median(valid_rewards)
      # print ('Iteration = {}, training loss = {}, average_validation_reward = {}'.format(i+1, curr_train_loss, valid_avg_rewards[-1]))
        if median_reward >= best_valid_reward:
          print('Best policy yet, saving')
          self.policy = deepcopy(policy_curr)
          self.policy.model.print_parameters()
          best_valid_reward = median_reward
        else: print('Keeping old policy')
        validation_reward[i,:]   = valid_rewards
        validation_accuracy[i,:] = valid_accs
    return train_rewards, train_loss, train_accs, validation_reward, validation_accuracy, dataset_size, weights_per_iter, features_per_iter, labels_per_iter


  def test(self, env, policy, num_episodes, render=False, step=False, quad_ftrs=False, dump_folder=None):
    test_rewards = {}
    test_acc = {}
    _, _ = env.reset(roll_back=True)
    policy.model.eval() #Put the model in eval mode to turn off dropout etc.
    policy.model.cpu() #Copy over the model to cpu if on cuda 
    with torch.no_grad(): #Turn off autograd
      for i in range(num_episodes):
        j = 0
        ep_reward = 0
        ep_acc = 0.0
        obs, _ = env.reset()
        self.G.reset()
        if render:
          path = self.get_path(self.G)
          feas_actions = self.filter_path(path, obs)
          ftrs = torch.tensor(self.G.get_features(feas_actions, obs, j, quad_ftrs))
          scores = policy.predict(ftrs)
          scores = scores.detach().numpy()
          # self.render_env(env, path, scores)
          act_id = feas_actions[np.argmax(scores)]
          self.render_env(env, self.G, obs, path, act_id, dump_folder)
        done = False
        # print('Curr episode = {}'.format(i+1))
        while not done:
          path = self.get_path(self.G)
          feas_actions = self.filter_path(path, obs)
          ftrs = torch.tensor(self.G.get_features(feas_actions, obs, j, quad_ftrs))
          scores = policy.predict(ftrs).detach().numpy()
          #select greedy action
          idx = np.random.choice(np.flatnonzero(scores==scores.max())) #random tie breaking
          idx_exp = self.expert(feas_actions, obs, j, self.G, env)
          if idx == idx_exp: ep_acc = ep_acc + 1.0

          act_id = feas_actions[idx]
          if render:
            # self.render_env(env, feas_actions, scores)
            self.render_env(env, self.G, obs, path, act_id, dump_folder=dump_folder)

          if step: raw_input('Press enter to execute action')
          obs, reward, done, info = env.step(act_id)
          self.G.update_edge(act_id, obs[act_id])#Update the edge weight according to the result
                  
          ep_reward += reward
          j += 1
        # print('Final path = {}'.format([env.edge_from_action(a) for a in path]))
        if render:
          fr = torch.tensor(self.G.get_features(path, obs, j, quad_ftrs))
          sc = policy.predict(fr)
          # self.render_env(env, path, sc.detach().numpy())
          self.render_env(env, self.G, obs, path, dump_folder=dump_folder)
          raw_input('Press Enter')
      
        test_rewards[env.world_num] = ep_reward
        test_acc[env.world_num] = ep_acc/(j*1.0)
        if step: raw_input('Episode over, press enter for next episode')

    return test_rewards, test_acc
  

  # def render_env(self, env, path, scores):
  #   # scores=scores.numpy()
  #   # edge_widths={}
  #   # edge_colors={}
  #   # for k,i in enumerate(path):
  #   #   edge_widths[i] = 5.0
      
  #   #   # print color_val
  #   # env.render(edge_widths=edge_widths, edge_colors=edge_colors)

  #   edge_widths={}
  #   edge_colors={}
  #   posts = self.G.get_posterior(range(len(obs)), obs)
  #   for i in range(len(obs)):
  #     edge_widths[i] = posts[i] + 0.15
    
  #   for i in path:
  #     edge_widths[i] = 5.0
  #     # edge_colors[i] = str(0.4)
  #     color_val = 1.0 - (scores[k][0] - np.min(scores))/(np.max(scores) - np.min(scores))
  #     color_val = max(min(color_val, 0.65), 0.1)
  #     edge_colors[i] = str(color_val)
  #     if i == act_id:
  #       edge_widths[i] = 7.0
  #       edge_colors[i] = str(0.0)
  #   self.env.render(edge_widths=edge_widths, edge_colors=edge_colors)

  
  def render_env(self, env, G, obs, path, act_id=-1, dump_folder=None):
    edge_widths={}
    edge_colors={}
    # posts = G.get_posterior(range(len(obs)), obs)
    # for i in range(len(obs)):
      # edge_widths[i] = posts[i] + 0.15
    for i in path:
      edge_widths[i] = 5.0
      edge_colors[i] = str(0.4)
      if i == act_id:
        edge_widths[i] = 7.0
        edge_colors[i] = str(0.0)
    env.render(edge_widths=edge_widths, edge_colors=edge_colors, dump_folder=dump_folder, curr_sp_len = G.curr_sp_len)  

  def beta_schedule(self, decay_episodes, eps0, epsf):
    sc = np.linspace(epsf, eps0, decay_episodes)
    return sc[::-1]
  
  # def select_prior(self, feas_actions, G):
  #   "Choose the edge most likely to be in collision"
  #   edges = list(map(self.train_env.edge_from_action, feas_actions))
  #   priors = G.get_priors(edges)
  #   return np.argmax(priors)

  def get_random_action(self, feas_actions, G):
    idx = np.random.randint(len(feas_actions)) #select purely random action
    return idx