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
  def __init__(self, train_env, valid_env, policy, beta0, expert_str, G, epochs):
    super(DaggerAgent, self).__init__(train_env)
    self.EXPERTS = {'select_expand': None,
                    'select_forward':   self.select_forward,
                    'select_backward':  self.select_backward,
                    'select_alternate': self.select_alternate,
                    'select_prior':     self.select_prior,
                    'select_posterior': self.select_posterior,
                    'select_ksp_centrality': self.select_ksp_centrality,
                    'select_posteriorksp':   self.select_posteriorksp,
                    'select_delta_len':  self.select_delta_len,
                    'select_delta_prog': self.select_delta_prog,
                    'select_posterior_delta_len': self.select_posterior_delta_len,
                    'length_oracle': self.length_oracle,
                    'length_oracle_2': self.length_oracle_2}
    
    self.train_env = train_env
    self.valid_env = valid_env
    self.expert = self.EXPERTS[expert_str]
    self.beta0 = beta0
    self.G = G
    self.train_epochs = epochs
    self.policy = policy


  def train(self, num_iterations, num_eval_episodes, num_valid_episodes, heuristic = None, re_init=False, render=False, step=False, mixed_rollin=False):
    train_rewards = []
    train_avg_rewards = []
    train_loss = []
    train_avg_loss = []
    validation_reward = []
    validation_std = []
    validation_avg_reward = []
    dataset_size_per_iter = []
    state_dict_per_iter = {}
    features_per_iter = {}
    labels_per_iter = {}

    D = ExperienceBuffer() 
    policy_curr = deepcopy(self.policy)
    best_valid_reward = -np.inf
    curr_median_reward = -np.inf
    for i in xrange(num_iterations):
      iter_rewards = []
      Xdata = []
      Ydata = []
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
          ep_reward = 0
          obs, _ = self.train_env.reset()
          self.G.reset()
          # probs = probs.detach().numpy()
          if render:
            path  = self.get_path(self.G)
            ftrs  = torch.tensor(self.G.get_features(feas_actions, obs, j))
            probs = policy_curr.predict(ftrs).detach().numpy()
            self.render_env(self.train_env, path, probs)

          done = False          
          while not done:
            print('Current iteration = {}, Iter episode = {}, Timestep = {}, beta = {}, Curr median reward = {}'.format(i+1, k+1, j, beta, curr_median_reward))
            path = self.get_path(self.G)
            feas_actions = self.filter_path(path, obs) #Only select from unevaluated edges in shortest path
            ftrs = torch.tensor(self.G.get_features(feas_actions, obs, j))
            probs = policy_curr.predict(ftrs)  
            ftrsl = self.G.get_features(feas_actions, obs, j).tolist()
            #Do random tie breaking
            probs = probs.detach().numpy()
            # print feas_actions, probs, ftrs
            idx_l = np.random.choice(np.flatnonzero(probs==probs.max())) #random tie breaking
            idx_exp = self.expert(feas_actions, obs, j, self.G)
            #Aggregate the data
            D.push(ftrs, torch.tensor([idx_exp]))            
            for id in xrange(len(feas_actions)):
              Xdata.append(ftrsl[id])
              if id == idx_exp:
                Ydata.append(1)
              else:
                Ydata.append(0)
            #Execute mixture
            if np.random.sample() > beta:
              idx = idx_l 
              # print('Learner action')
            else:
              idx = idx_exp
              if heuristic is not None: 
                idx_heuristic = self.EXPERTS[heuristic](feas_actions, obs, j, self.G) 
                if mixed_rollin:
                  idx = np.random.choice([idx_exp, idx_heuristic])
                else:
                  idx = idx_heuristic
              if idx   == idx_exp: print('Expert action')
              elif idx == idx_heuristic: print('Heuristic action') 
            
            act_id = feas_actions[idx]
            # act_e = self.train_env.edge_from_action(act_id)
            # print ('Q_vals = {}, feasible_actions = {}, chosen edge = {}, features = {}'.format(q_vals, feas_actions, act_e, ftrs))
            obs,reward, done, info = self.train_env.step(act_id)
            self.G.update_edge(act_id, obs[act_id]) #Update agent's internal graph          
            j += 1
            ep_reward += reward
          
          iter_rewards.append(ep_reward)
          train_rewards.append(ep_reward)
          train_avg_rewards.append(sum(train_rewards)*1.0/(i*num_eval_episodes + k +1.0))      
        curr_median_reward = np.median(iter_rewards)
      #re-initialize policy and train
      if re_init:
        print('Current policy parameters')
        policy_curr.model.print_parameters()
        policy_curr.model.apply(init_weights)
        print('Re-initialized policy parameters')
        policy_curr.model.print_parameters()
      
      
      curr_train_loss = policy_curr.train(D, self.train_epochs)
      dataset_size_per_iter.append(len(D))
      state_dict_per_iter[i] = deepcopy(policy_curr.model.state_dict())
      features_per_iter[i] = Xdata
      labels_per_iter[i] = Ydata
      # print('Running validation')
      #Doing validation if num_valid episodes > 0
      if num_valid_episodes == 0:
        self.policy = deepcopy(policy_curr)
        median_reward = 0
        reward_std = 0
      else:
        valid_rewards_dict, valid_avg_rewards_dict = self.test(self.valid_env, policy_curr, num_valid_episodes)
        valid_rewards     = [it[1] for it in sorted(valid_rewards_dict.items(), key=operator.itemgetter(0))]
        valid_avg_rewards = [it[1] for it in sorted(valid_avg_rewards_dict.items(), key=operator.itemgetter(0))]        
        median_reward = np.median(valid_rewards)
        reward_std = np.std(valid_rewards)
      # print ('Iteration = {}, training loss = {}, average_validation_reward = {}'.format(i+1, curr_train_loss, valid_avg_rewards[-1]))
        if median_reward >= best_valid_reward:
          # print('Best policy yet, saving')
          self.policy = deepcopy(policy_curr)
          best_valid_reward = median_reward

      # print 'Average validation reward = {}, std = {}, best reward yet = {}'.format(median, reward_std, best_valid_reward)      
      # raw_input('...')
      validation_reward.append(median_reward)
      validation_std.append(reward_std)
      validation_avg_reward.append(sum(validation_reward)*1.0/(i+1.0))
      train_loss.append(curr_train_loss)
      train_avg_loss.append(sum(train_loss)*1.0/(i+1.0))
    # print self.qfun.model.fc.weight
    return train_rewards, train_avg_rewards, train_loss, train_avg_loss, validation_reward, validation_std, validation_avg_reward, dataset_size_per_iter, state_dict_per_iter, features_per_iter, labels_per_iter


  def test(self, env, policy, num_episodes, render=False, step=False):
    # print policy.model.fc.weight, policy.model.fc.bias
    test_rewards = {}
    test_avg_rewards = {}
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
        path = self.get_path(self.G)
        ftrs = torch.tensor(self.G.get_features([env.edge_from_action(a) for a in path], j))
        probs = policy.predict(ftrs)
        probs = probs.detach().numpy()
        if render:
          self.render_env(env, path, probs)
        done = False
        # print('Curr episode = {}'.format(i+1))
        while not done:
          path = self.get_path(self.G)
          feas_actions = self.filter_path(path, obs)
          ftrs = torch.tensor(self.G.get_features([env.edge_from_action(a) for a in feas_actions], j))
          probs = policy.predict(ftrs)
          probs = probs.detach().numpy()
          if render:
            self.render_env(env, feas_actions, probs)
          #select greedy action
          idx = np.random.choice(np.flatnonzero(probs==probs.max())) #random tie breaking
          idx_exp = self.expert(feas_actions, j, self.train_env, self.G)
          if idx == idx_exp: ep_acc = ep_acc + 1.0

          act_id = feas_actions[idx]
          act_e = env.edge_from_action(act_id)
          # print q_vals[idx], ftrs[idx]
          # print ('Q_vals = {}, feasible_actions = {}, chosen edge = {}, features = {}'.format(q_vals, feas_actions, act_e, ftrs))
          if step: raw_input('Press enter to execute action')
          obs, reward, done, info = env.step(act_id)
          # print ('obs = {}, reward = {}, done = {}'.format(obs, reward, done))    
          self.G.update_edge(act_e, obs[act_id])#Update the edge weight according to the result
                  
          ep_reward += reward
          j += 1
        # print(ep_reward)
        # print('Final path = {}'.format([env.edge_from_action(a) for a in path]))
        #Render environment one last time
        
        if render:
          fr = torch.tensor(self.G.get_features([env.edge_from_action(a) for a in path], j))
          sc = policy.predict(fr)
          self.render_env(env, path, sc.detach().numpy())
          raw_input('Press Enter')
        

        test_rewards[env.world_num] = ep_reward
        # print env.world_num
        test_acc[env.world_num] = ep_acc/(j*1.0)
        test_avg_rewards[env.world_num] = np.mean(test_rewards.values())
        # test_rewards.append(ep_reward)
        # test_avg_rewards.append(sum(test_rewards)*1.0/(i+1.0))
        if step: raw_input('Episode over, press enter for next episode')

    return test_rewards, test_avg_rewards, test_acc
  

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