#!/usr/bin/env python
import os,sys
sys.path.insert(0, os.path.abspath('..'))
import matplotlib.pyplot as plt
import numpy as np
import gym
import sp_rl
from sp_rl.agents import Graph
import json



def get_path(env, graph):
  sp = graph.curr_shortest_path
  sp = env.to_edge_path(sp) 
  return sp

def filter_path(path, obs):
  """This function filters out edges that have already been evaluated"""
  path_f = []
  for i in path:
    if obs[i] == -1:
      path_f.append(i)
  return path_f

def select_prior(feas_actions, env, G):
  "Choose the edge most likely to be in collision"
  edges = list(map(env.edge_from_action, feas_actions))
  priors = G.get_priors(edges)
  idx_prior = np.argmax(priors)
  return idx_prior


def length_oracle(feas_actions, env, G, horizon=2):
  curr_sp = G.curr_shortest_path
  curr_sp_len = G.curr_shortest_path_len
  scores = [0.0]*len(feas_actions)
  for (j, action) in enumerate(feas_actions):
    edge = env.edge_from_action(action)
    if env.G[edge[0]][edge[1]]['status'] == 0:
      G.update_edge(edge, 0)
      new_sp = G.curr_shortest_path
      new_sp_len = G.curr_shortest_path_len
      scores[j] = scores[j] + (new_sp_len-curr_sp_len)
      G.update_edge(edge, -1)
  return np.argmax(scores)

def get_score(feas_actions, env, G, horizon=2):
  curr_sp = G.curr_shortest_path
  print(G.in_edge_form(curr_sp))
  curr_sp_len = G.curr_shortest_path_len
  scores = [0.0]*len(feas_actions)
  for (j, action) in enumerate(feas_actions):
    edge = env.edge_from_action(action)
    if env.G[edge[0]][edge[1]]['status'] == 0:
      G.update_edge(edge, 0)
      new_sp = G.curr_shortest_path
      new_sp_len = G.curr_shortest_path_len
      scores[j] = scores[j] + (new_sp_len-curr_sp_len)
      G.update_edge(edge, -1)
  return scores

def get_prior(feas_actions, env, G):
  "Choose the edge most likely to be in collision"
  edges = list(map(env.edge_from_action, feas_actions))
  priors = G.get_priors(edges)
  return priors


def render_env(env, obs, path, act_id=-1):
  edge_widths={}
  edge_colors={}
  for i in path:
    edge_widths[i] = 4.0
    edge_colors[i] = str(0.4)
    if i == act_id:
      edge_widths[i] = 7.0
      edge_colors[i] = str(0.0)
  env.render(edge_widths=edge_widths, edge_colors=edge_colors)

def main(args):
  env1 = gym.make(args.env)
  env2 = gym.make(args.env)
  env1.seed(0)
  env2.seed(0)
  np.random.seed(0)
  _, graph_info1 = env1.reset()
  _, graph_info2 = env2.reset()
  ftr_params=None
  G1 = Graph(graph_info1['adj_mat'], graph_info1['source_node'], graph_info1['target_node'], ftr_params=ftr_params, pos=graph_info1['pos'], edge_priors=graph_info1['edge_priors'], lite_ftrs=True)
  G2 = Graph(graph_info2['adj_mat'], graph_info2['source_node'], graph_info2['target_node'], ftr_params=ftr_params, pos=graph_info2['pos'], edge_priors=graph_info2['edge_priors'], lite_ftrs=True)

  
  obs1, _ = env1.reset()
  obs2, _ = env2.reset()
  j = 0
  done = False
  ep_reward1 = 0
  ep_reward2 = 0
  while not done:
    path1 = get_path(env1, G1)
    path2 = get_path(env2, G2)
    feas_actions1 = filter_path(path1, obs1)
    feas_actions2 = filter_path(path2, obs2)
    scores1 = get_score(feas_actions1, env1, G1)
    scores2 = get_score(feas_actions2, env2, G2)
    priors1 = get_prior(feas_actions1, env1, G1)
    priors2 = get_prior(feas_actions2, env2, G2)
    # print path, feas_actions
    
    act_id1 = select_prior(feas_actions1, env1, G1)
    act_id2 = length_oracle(feas_actions2, env2, G2)
    act_e1 = env1.edge_from_action(feas_actions1[act_id1])
    act_e2 = env2.edge_from_action(feas_actions2[act_id2])
    print('select_prior. feas actions = {}, edge selected = {}, score = {}, prior = {}'.format(feas_actions1, act_e1, scores1[act_id1], priors1[act_id1]))    
    print('length_oracle. feas actions = {}, edge selected = {}, score = {}, prior = {}'.format(feas_actions2, act_e2, scores2[act_id2], priors2[act_id2]))    
    if args.render:
      render_env(env1, obs1, feas_actions1, act_id1)
      render_env(env2, obs2, feas_actions2, act_id2)

    # print('feasible actions = {}, chosen edge = {}, edge_features = {}'.format(feas_actions, [act_id, act_e], ftrs))
    if args.step: raw_input('Press enter to execute action')
    obs1, reward1, done1, info1 = env1.step(feas_actions1[act_id1])
    obs2, reward2, done2, info2 = env2.step(feas_actions2[act_id2])
    # print ('obs = {}, reward = {}, done = {}'.format(obs, reward, done))
     
    G1.update_edge(act_e1, obs1[feas_actions1[act_id1]])#Update the edge weight according to the result
    G2.update_edge(act_e2, obs2[feas_actions2[act_id2]])#Update the edge weight according to the result
    ep_reward1 += reward1
    ep_reward2 += reward2
    if done1 and done2: done = True
    j += 1



if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Policy that heuristically selects edges to expand')
  parser.add_argument('--env', type=str, required=True, help='environment name')
  parser.add_argument('--render', action='store_true',
                      help='Render during testing')
  parser.add_argument("--step", action='store_true', help='If true, then agent pauses before executing each action to show rendering')
  args = parser.parse_args()
  main(args)
