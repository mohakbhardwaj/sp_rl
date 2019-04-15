#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import gym
import sp_rl
import networkx as nx
from agent import Agent
from graph_wrapper import GraphWrapper
import time
import csv

class HeuristicAgent(Agent):
  def __init__(self, env, selector_str, ftr_params=None, lite_ftrs=False):
    super(HeuristicAgent, self).__init__(env)
    self.SELECTORS = {'select_expand': None,
                      'select_forward': self.select_forward,
                      'select_backward': self.select_backward,
                      'select_alternate': self.select_alternate,
                      'select_prior': self.select_prior,
                      'select_posterior': self.select_posterior,
                      'select_ksp_centrality': self.select_ksp_centrality,
                      'select_multiple': self.select_multiple,
                      'select_priorkshort': self.select_priorkshort,
                      'select_delta_len': self.select_delta_len,
                      'select_delta_prog': self.select_delta_prog,
                      'select_posterior_delta_len': self.select_posterior_delta_len,
                      'length_oracle':self.length_oracle,
                      'length_oracle_2':self.length_oracle_2}

    self.selector_str = selector_str
    self.selector = self.SELECTORS[selector_str]
    _, self.graph_info = self.env.reset(roll_back=True)
    self.ftr_params = ftr_params
    if 'pos' in self.graph_info: node_pos = self.graph_info['pos']
    else: node_pos = None
    self.G = GraphWrapper(self.graph_info)


  def train(self, num_episodes, render=False, step=False):
    return self.test(num_episodes, render, step)
    

  def test(self, num_episodes, render=True, step=False):
    start_t = time.time()
    test_rewards = {}
    test_avg_rewards = {}
    for i in range(num_episodes):
      sp_len_arr = []
      edge_check_arr = []
      invalid_check_arr = []
      posterior_arr = []
      delta_len_arr = []
      obs, _ = self.env.reset()
      self.G.reset()
      path = self.get_path(self.G)
      if render:
        self.render_env(obs, path)
      j = 0
      k = 0
      run_base = False 
      ep_reward = 0
      done = False
      while not done:
        path = self.get_path(self.G)
        feas_actions = self.filter_path(path, obs)
        act_id = self.selector(feas_actions, obs, j+1, self.G)
        # act_e = self.env.edge_from_action(act_id)
        sp_len_arr.append(self.G.curr_sp_len)
        edge_check_arr.append(j)
        invalid_check_arr.append(k)
        post = self.G.get_posterior([act_id], obs)
        posterior_arr.append(post[0])
        delta_len = self.G.get_delta_len_util([act_id])
        delta_len_arr.append(delta_len[0])

        if render:
          self.render_env(obs, feas_actions, act_id)
        # print('feasible actions = {}, chosen edge = {}, edge_features = {}'.format(feas_actions, [act_id, act_e], ftrs))
        if step: raw_input('Press enter to execute action')
        obs, reward, done, info = self.env.step(act_id)
        # print path, feas_actions, obs[act_id]
        # print ('obs = {}, reward = {}, done = {}'.format(obs, reward, done))
         
        self.G.update_edge(act_id, obs[act_id]) #Update the edge weight according to the result
        ep_reward += reward
        j += 1
        if obs[act_id] == 0: k += 1

      #Render environment one last time
      if render:
        self.render_env(obs, path)
        raw_input('Press Enter')
        
      test_rewards[self.env.world_num] = ep_reward
      test_avg_rewards[self.env.world_num] = np.mean(test_rewards.values())
      # plt.figure()
      # plt.plot(sp_len_arr, edge_check_arr,    label='Total edges checked')
      # plt.plot(sp_len_arr, invalid_check_arr, label='Invalid edges checked')
      results_dict = {}
      results_dict['sp_len'] = sp_len_arr
      results_dict['edge_checked'] = edge_check_arr
      results_dict['invalid_checked'] = invalid_check_arr 
      results_dict['posterior'] = posterior_arr
      results_dict['delta_len'] = delta_len_arr
      with open(self.selector_str + '_data.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, dialect=csv.excel)
        for key, value in results_dict.items():
          row=[key]
          for v in value: row.append(v)
          writer.writerow(row)
      if step: raw_input('Episode over, press enter for next episode')

      # plt.xlabel('Current shortest path length')  
      # plt.ylabel('Edge checks')
      # plt.legend()
      # plt.show()  
    # with open('posterior_plot.txt', 'w') as f:
    #   for item in my_list:
    #     f.write("%s\n" % item)

    avg_time_taken = time.time() - start_t
    avg_time_taken = avg_time_taken/num_episodes*1.0
    
    return test_rewards, test_avg_rewards, avg_time_taken


  
  def render_env(self, obs, path, act_id=-1):
    edge_widths={}
    edge_colors={}
    for i in path:
      edge_widths[i] = 5.0
      edge_colors[i] = str(0.4)
      if i == act_id:
        edge_widths[i] = 7.0
        edge_colors[i] = str(0.0)
    self.env.render(edge_widths=edge_widths, edge_colors=edge_colors)
  
  def select_forward(self, act_ids, obs, iter, G):
    return act_ids[0]

  def select_backward(self, act_ids, obs, iter, G):
    return act_ids[-1]

  def select_alternate(self, act_ids, obs, iter, G):
    if iter%2 == 0:
      return act_ids[-1]
    else:
      return act_ids[0]

  def select_prior(self, act_ids, obs, iter, G):
    "Choose the edge most likely to be in collision"
    priors = G.get_priors(act_ids)
    idx_prior = np.argmax(priors)
    # idx_prior = np.random.choice(np.flatnonzero(priors == priors.max()))
    return act_ids[idx_prior]

  def select_posterior(self, act_ids, obs, iter, G):
    "Choose the edge most likely to be in collision given all the observations so far"
    post = G.get_posterior(act_ids, obs)
    idx_post = np.argmax(post)
    # idx_post = np.random.choice(np.flatnonzero(post == post.max()))
    return act_ids[idx_post]

  def select_delta_len(self, act_ids, obs,iter, G):
    # edges = list(map(self.env.edge_from_action, act_ids))
    delta_lens = G.get_delta_len_util(act_ids)
    idx_lens = np.argmax(delta_lens)#np.random.choice(np.flatnonzero(delta_lens == delta_lens.max()))
    return act_ids[idx_lens]

  def select_delta_prog(self, act_ids, obs, iter, G):
    delta_progs = G.get_delta_prog_util(act_ids)
    idx_prog = np.argmax(delta_progs)#np.random.choice(np.flatnonzero(delta_progs == delta_progs.max()))
    return act_ids[idx_prog]

  def select_posterior_delta_len(self, act_ids, obs, iter, G):
    p = G.get_posterior(act_ids, obs)
    dl = G.get_delta_len_util(act_ids)
    pdl = p * dl
    idx_pdl = np.argmax(pdl)
    return act_ids[idx_pdl] 


  def select_ksp_centrality(self, act_ids, obs, iter, G):
    ksp_centr = G.get_ksp_centrality(act_ids)
    print ksp_centr
    return act_ids[np.argmax(ksp_centr)]

  def select_multiple(self, act_ids, obs, iter, G):
    """Choose the best edge according to all heuristics"""
    edges = list(map(self.env.edge_from_action, act_ids))
    ftrs = G.get_features(edges, iter) 
    h = np.min(ftrs, axis=1)
    return act_ids[np.argmax(h)]

  def select_priorkshort(self, act_ids, obs, iter, G):
    edges = list(map(self.env.edge_from_action, act_ids))
    priors = G.get_priors(edges)
    kshort = [num*1.0/self.ftr_params['k']*1. for num in G.get_k_short_num(edges)]
    h = map(lambda x: x[0]*x[1], zip(kshort,priors))
    return act_ids[np.argmax(h)]
  


  def length_oracle(self, act_ids, obs, iter, G, horizon=2):
    curr_sp = G.curr_shortest_path
    curr_sp_len = G.curr_shortest_path_len
    scores = np.array([0.0]*len(act_ids))
    for (j, action) in enumerate(act_ids):
      gt_edge = self.env.gt_edge_from_action(action)
      edge = (gt_edge.source(), gt_edge.target())
      if self.env.G.edge_properties['status'][gt_edge] == 0:
        G.update_edge(edge, 0)
        new_sp = G.curr_shortest_path
        new_sp_len = G.curr_shortest_path_len
        scores[j] = scores[j] + (new_sp_len-curr_sp_len)
        G.update_edge(edge, -1, 0)
    action = act_ids[np.argmax(scores)]
    return action

  def length_oracle_2(self, act_ids, obs, iter, G, horizon=2):
    curr_sp = G.curr_shortest_path
    curr_sp_len = G.curr_shortest_path_len
    scores = np.array([0.0]*len(act_ids))
    
    for (j, action) in enumerate(act_ids):
      gt_edge = self.env.gt_edge_from_action(action)
      edge = (gt_edge.source(), gt_edge.target())
      if self.env.G.edge_properties['status'][gt_edge] == 0:
        G.update_edge(edge, 0)
        new_sp = G.curr_shortest_path
        new_sp_len = G.curr_shortest_path_len
        reward = (new_sp_len - curr_sp_len)

        edge_sp = G.in_edge_tup_form(new_sp)
        f_a = [self.env.action_from_edge(e) for e in edge_sp]
        value = np.max(self.get_scores(f_a, iter, G))
        scores[j] = reward + value
        G.update_edge(edge, -1, 0)
    
    action = act_ids[np.argmax(scores)]
    return action



  def get_scores(self, feas_actions, iter, G):
    curr_sp = G.curr_shortest_path
    curr_sp_len = G.curr_shortest_path_len
    scores = [0.0]*len(feas_actions)
    for (j, action) in enumerate(feas_actions):
      # edge = self.env.edge_from_action(action)
      gt_edge = self.env.gt_edge_from_action(action)
      edge = (gt_edge.source(), gt_edge.target())
      if self.env.G.edge_properties['status'][gt_edge] == 0:
        G.update_edge(edge, 0)
        new_sp = G.curr_shortest_path
        if len(new_sp) > 0:
          new_sp_len = G.curr_shortest_path_len
          reward = new_sp_len-curr_sp_len
        else: reward = np.inf
        scores[j] = reward
        G.update_edge(edge, -1)
    return scores