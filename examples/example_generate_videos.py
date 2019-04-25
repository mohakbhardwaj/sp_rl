#!/usr/bin/env python
"""Load edge ids from files and run in worlds"""
import os,sys
sys.path.insert(0, os.path.abspath('..'))
import operator
import matplotlib.pyplot as plt
import numpy as np
import gym
import time
import sp_rl
from sp_rl.agents import GraphWrapper 
import json
import csv

def get_path(env, graph):
  sp = graph.curr_shortest_path
  sp = env.to_edge_path(sp) 
  return sp

def filter_path(path, obs):
  """This function filters out edges that have already been evaluated"""
  path_f = list(filter(lambda x: obs[x] == -1, path))
  return path_f

def render_env(env, obs, path, act_id=-1):
  edge_widths={}
  edge_colors={}
  for i in path:
    edge_widths[i] = 5.0
    edge_colors[i] = str(0.4)
    if i == act_id:
      edge_widths[i] = 7.0
      edge_colors[i] = str(0.0)
  env.render(edge_widths=edge_widths, edge_colors=edge_colors)


def main(args):
  env = gym.make(args.env)
  env.seed(args.seed_val)
  np.random.seed(args.seed_val)
  _, graph_info = env.reset(roll_back=True)
  G = GraphWrapper(graph_info, args.k)

  for file in args.files:
    print('File name = {}'.format(file))
    action_dict = {}
    with open(file, 'r') as fp:
      csv_reader = csv.reader(fp, delimiter=',')
      for row in csv_reader:
        print row
        action_dict[int(row[0])] = [int(i) for i in row[1:]]
    print action_dict

    start_t = time.time()
    for i in range(args.num_episodes):
      obs, _ = env.reset()
      G.reset()
      if args.render:
        path = get_path(env, G)
        render_env(env, obs, path)
        raw_input('Press enter to start')
      j = 0
      k = 0
      run_base = False 
      ep_reward = 0
      ep_edge_ids = []
      done = False

      while not done:
        path = get_path(env, G)
        feas_actions = filter_path(path, obs)
        act_id = action_dict[env.world_num][j]
        if args.render:
          render_env(env, obs, feas_actions, act_id)
        if args.step: raw_input('Press enter to execute action')
        obs, reward, done, info = env.step(act_id)         
        G.update_edge(act_id, obs[act_id]) #Update the edge weight according to the result
        ep_reward += reward
        ep_edge_ids.append(act_id)
        j += 1
        if obs[act_id] == 0: k += 1

      #Render environment one last time
      if args.render:
        render_env(env, obs, path)
        raw_input('Press Enter')
      
      if args.step: raw_input('Episode over, press enter for next episode')

    avg_time_taken = time.time() - start_t
    avg_time_taken = avg_time_taken/num_episodes*1.0
    



if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Policy that heuristically selects edges to expand')
  parser.add_argument('--env', type=str, required=True, help='environment name')
  parser.add_argument('--files', type=str, nargs='+', required=True)
  parser.add_argument('--num_episodes', type=int, required=True)
  parser.add_argument('--k', type=int)
  parser.add_argument('--render', action='store_true', help='Render during testing')
  parser.add_argument("--step", action='store_true', help='If true, then agent pauses before executing each action to show rendering')
  parser.add_argument("--seed_val", type=int, required=True, help='Seed value')
  args = parser.parse_args()
  main(args)
