#!/usr/bin/env python
import os, sys
sys.path.insert(0, os.path.abspath(".."))
import matplotlib.pyplot as plt 
import gym
import sp_rl
import networkx as nx
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Random policy')
parser.add_argument('--env', type=str, required=True, help='environment name')
parser.add_argument('--num_episodes', type=int, required=True)
parser.add_argument('--render', action='store_true')
args = parser.parse_args()
num_episodes = args.num_episodes
tot_reward = 0.0
env = gym.make(args.env)
env.seed(0)
np.random.seed(0)
_, graph_info = env.reset()

rewards = [] #Reward per episode
avg_rewards = [] #Running average of rewards across episodes

def get_path(graph, graph_info, obs):
  sp = nx.shortest_path(graph, graph_info['source_node'], graph_info['target_node'], 'weight')
  sp = env.to_edge_path(sp) 
  return sp


def filter_path(path, obs):
  """This function filters out edges that have already been evaluated"""
  path_f = []
  for i in path:
    if obs[i] == -1:
      path_f.append(i)
  return path_f


for i in range(num_episodes):
  obs, graph_info = env.reset()
  G = nx.from_numpy_matrix(graph_info['adj_mat'])
  if args.render:
    env.render()
  ep_reward = 0
  #Run an episode with a random policy
  done = False
  while not done:
    path = get_path(G, graph_info, obs)
    feas_actions = filter_path(path, obs)
    print('Curr episode = {}'.format(i+1))
    act_id = np.random.choice(feas_actions)
    act_e = env.edge_from_action(act_id)
    obs, reward, done, info = env.step(act_id)
    # print ('chosen edge = {}, obs = {}, reward = {}, done = {}'.format(act_e, obs, reward, done))
    if args.render:
      env.render()
    if obs[act_id] == 0: G[act_e[0]][act_e[1]]['weight'] = np.inf
    ep_reward += reward
  rewards.append(ep_reward)
  avg_rewards.append(sum(rewards)*1.0/(i+1.0))

print('Number of episodes = {}, Total Reward = {}, Average reward = {}'.format(num_episodes, sum(rewards), sum(rewards)*1.0/num_episodes*1.))

fig, ax = plt.subplots(2,1)
ax[0].plot(range(num_episodes), rewards)
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Episode Reward')

ax[1].plot(range(num_episodes), avg_rewards)
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Average Reward')

plt.show()
