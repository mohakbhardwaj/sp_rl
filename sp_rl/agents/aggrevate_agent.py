#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import sp_rl
import networkx as nx
from agent import Agent
from graph import Graph
from copy import deepcopy
from sp_rl.learning import LinearQFunction, ExperienceBuffer, select_forward, select_backward, select_alternate, select_prior, select_lookahead

class AggrevateAgent(Agent):
  def __init__(self, train_env, valid_env, qfun, beta0, T, expert_str, G, epochs):
    super(AggrevateAgent, self).__init__(train_env)
    self.EXPERTS = {'select_expand': None,
                    'select_forward': select_forward,
                    'select_backward': select_backward,
                    'select_alternate': select_alternate,
                    'select_prior': select_prior, 
                    'select_lookahead': select_lookahead}
    self.train_env = train_env
    self.valid_env = valid_env
    self.expert = self.EXPERTS[expert_str]
    self.beta0 = beta0
    self.G = G
    self.train_epochs = epochs
    self.qfun = qfun
    self.T = T


  def train(self, num_iterations, num_eval_episodes, num_valid_episodes, use_advantage=True, render=False, step=False):
    train_rewards = []
    train_avg_rewards = []
    train_loss = []
    train_avg_loss = []
    validation_reward = []
    validation_avg_reward = []
    dataset_size_per_iter = []
    D = ExperienceBuffer() 
    qfun_curr = deepcopy(self.qfun)
    best_valid_reward = -np.inf

    for i in xrange(num_iterations):
      if i == 0:
        beta = 1.0 #Rollout with expert initially
      else:
        beta = self.beta0/i*1.0
      # beta = 1.0

      #Evaluate every policy for k episodes
      for k in xrange(num_eval_episodes):
        j = 0
        ep_reward = 0
        T = np.random.randint(0, self.T)

        obs, _ = self.train_env.reset()
        self.G.reset()
        path = self.get_path(self.G)
        ftrs = torch.tensor(self.G.get_features([self.train_env.edge_from_action(a) for a in path], j))
        q_vals = qfun_curr.predict(ftrs)
        if render:
          self.render_env(self.train_env, path, q_vals)

        done = False        
        #Data to be collected each episode
        state_data = None  #features of all actions that can be performed 
        action_data = None #features of selected action
        target_q = 0.0     #oracle q-value from roll-out
        target_v = 0.0     #oracle value target
        while not done:
          print('Current iteration = {}, Iter episode = {}, Timestep = {}, beta = {}, T = {}'.format(i+1, k+1, j, beta, T))
          path = self.get_path(self.G)
          feas_actions = self.filter_path(path, obs) #Only select from unevaluated edges in shortest path
          ftrs = torch.tensor(self.G.get_features([self.train_env.edge_from_action(a) for a in feas_actions], j))
          q_vals = qfun_curr.predict(ftrs)  
          idx_l = np.random.choice(np.flatnonzero(q_vals==q_vals.max()))
          idx_exp = self.expert(feas_actions, j, self.train_env, self.G)

          if j < T: #Roll in with mixture 
            if np.random.sample() > beta:
              idx = idx_l #random tie breaking
              # print('Rolling in ... learner action')
            else:
              idx = idx_exp
              # print('Rolling in ... expert action')
            act_id = feas_actions[idx] 
            act_e = self.train_env.edge_from_action(act_id)
            # print ('Q_vals = {}, feasible_actions = {}, chosen edge = {}, features = {}'.format(q_vals, feas_actions, act_e, ftrs))
            obs,reward, done, info = self.train_env.step(act_id)
            
            self.G.update_edge(act_e, obs[act_id]) #Update agent's internal graph          
            j += 1
            ep_reward += reward
          
          
          elif j == T:
            #branch out with expert to get state-value of current state
            if use_advantage:
              target_v = self.get_expert_value(deepcopy(self.train_env), deepcopy(self.G), deepcopy(obs), j)
            
            #Select a random action and store it's features
            idx = self.get_random_action(feas_actions, self.G)
            state_data = ftrs
            action_data = ftrs[idx,:].unsqueeze(1)
            act_id = feas_actions[idx] 
            act_e = self.train_env.edge_from_action(act_id)

            #Execute random action and get action-value wrt expert
            print ('Q_vals = {}, feasible_actions = {}, chosen edge = {}, features = {}'.format(q_vals, feas_actions, act_e, ftrs))
            obs,reward, done, info = self.train_env.step(act_id)
            self.G.update_edge(act_e, obs[act_id])
            target_q = 0
            if not done:
              target_q = reward + self.get_expert_value(deepcopy(self.train_env), deepcopy(self.G), deepcopy(obs), j+1)
            
            ep_reward += target_q
            break


        #Aggregate data to original dataset
        if state_data is not None and action_data is not None and target_q is not None:
          print target_q, target_v, float(target_q - target_v)
          target = float(target_q - target_v)
          D.push(action_data, torch.tensor([target]))

        train_rewards.append(ep_reward)
        train_avg_rewards.append(sum(train_rewards)*1.0/(i*num_eval_episodes + k +1.0))      
      
      dataset_size_per_iter.append(len(D))
      print('Training policy on aggregated data')
      curr_train_loss = qfun_curr.train(D, self.train_epochs)
      valid_rewards, valid_avg_rewards = self.test(qfun_curr, num_valid_episodes)
      
      # print ('Iteration = {}, training loss = {}, average_validation_reward = {}'.format(i+1, curr_train_loss, valid_avg_rewards[-1]))
      
      print valid_avg_rewards[-1], best_valid_reward
      if valid_avg_rewards[-1] >= best_valid_reward:
        print('Best policy yet, saving')
        self.qfun = deepcopy(qfun_curr)
        best_valid_reward = valid_avg_rewards[-1]

      validation_reward.append(valid_avg_rewards[-1])
      validation_avg_reward.append(sum(validation_reward)*1.0/(i+1.0))
      train_loss.append(curr_train_loss)
      train_avg_loss.append(sum(train_loss)*1.0/(i+1.0))
    # print self.qfun.model.fc.weight
    return train_rewards, train_avg_rewards, train_loss, train_avg_loss, validation_reward, validation_avg_reward, dataset_size_per_iter


  def get_expert_value(self, env, G, obs, j):
    tot_reward = 0
    done = False
    while not done:
      path = self.get_path(G)
      feas_actions = self.filter_path(path, obs)
      idx = self.expert(feas_actions, j, env, G)
      act_id = feas_actions[idx]
      act_e = env.edge_from_action(act_id)
      obs, reward, done, info = env.step(act_id)
      G.update_edge(act_e, obs[act_id])
      tot_reward = tot_reward + reward
      j = j + 1
    return tot_reward



  def test(self, qfun, num_episodes, render=False, step=False):
    # print qfun.model.fc.weight, qfun.model.fc.bias
    test_rewards = []
    test_avg_rewards = []
    with torch.no_grad():
      for i in range(num_episodes):
        j = 0
        ep_reward = 0
        obs, _ = self.valid_env.reset()
        self.G.reset()
        path = self.get_path(self.G)
        ftrs = torch.tensor(self.G.get_features([self.valid_env.edge_from_action(a) for a in path], j))
        q_vals = qfun.predict(ftrs)
        if render:
          self.render_env(self.valid_env, path, q_vals)
        done = False
        # print('Curr episode = {}'.format(i+1))
        while not done:
          path = self.get_path(self.G)
          feas_actions = self.filter_path(path, obs)
          ftrs = torch.tensor(self.G.get_features([self.valid_env.edge_from_action(a) for a in feas_actions], j))
          q_vals = qfun.predict(ftrs)
          if render:
            self.render_env(self.valid_env, feas_actions, q_vals)
          #select greedy action
          idx = np.random.choice(np.flatnonzero(q_vals==q_vals.max())) #random tie breaking
          act_id = feas_actions[idx]
          act_e = self.valid_env.edge_from_action(act_id)
          # print q_vals[idx], ftrs[idx]
          # print ('Q_vals = {}, feasible_actions = {}, chosen edge = {}, features = {}'.format(q_vals, feas_actions, act_e, ftrs))
          if step: raw_input('Press enter to execute action')
          obs, reward, done, info = self.valid_env.step(act_id)
          # print ('obs = {}, reward = {}, done = {}'.format(obs, reward, done))    
          self.G.update_edge(act_e, obs[act_id])#Update the edge weight according to the result
                  
          ep_reward += reward
          j += 1

        # print('Final path = {}'.format([self.valid_env.edge_from_action(a) for a in path]))
        #Render environment one last time
        if render:
          fr = torch.tensor(self.G.get_features([self.valid_env.edge_from_action(a) for a in path], j))
          qr = qfun.predict(fr)
          self.render_env(self.valid_env, path, qr)
          raw_input('Press Enter')

        test_rewards.append(ep_reward)
        test_avg_rewards.append(sum(test_rewards)*1.0/(i+1.0))
        if step: raw_input('Episode over, press enter for next episode')

    return test_rewards, test_avg_rewards
  

  def render_env(self, env, path, q_vals):
    q_vals=q_vals.numpy()
    edge_widths={}
    edge_colors={}
    for k,i in enumerate(path):
      edge_widths[i] = 5.0#min(max(abs(13.0/(q_vals[k] + 0.001)), 1.5),13.0)#, 5.0)
      color_val = 1.0 - (q_vals[k][0] - np.min(q_vals))/(np.max(q_vals) - np.min(q_vals))
      color_val = max(min(color_val, 0.65), 0.1)
      edge_colors[i] = str(color_val)
      # print color_val
    env.render(edge_widths=edge_widths, edge_colors=edge_colors)

  def beta_schedule(self, decay_episodes, eps0, epsf):
    sc = np.linspace(epsf, eps0, decay_episodes)
    return sc[::-1]
  

  def select_prior(self, feas_actions, G):
    "Choose the edge most likely to be in collision"
    edges = list(map(self.train_env.edge_from_action, feas_actions))
    priors = G.get_priors(edges)
    return np.argmax(priors)

  def get_random_action(self, feas_actions, G):
    # num = np.random.randint(0,4)
    # if num == 0: 
    #   idx = 0 #select forward
    # elif num == 1:
    #   idx = -1 #select backward
    # elif num == 2:
    #   idx = self.select_prior(feas_actions, G) #select most likely in collision
    # else:
    idx = np.random.randint(len(feas_actions)) #select purely random action
    # idx = 0
    return idx