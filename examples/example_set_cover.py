#!/usr/bin/env python
import os,sys
sys.path.insert(0, os.path.abspath('..'))
import matplotlib.pyplot as plt
import numpy as np
import gym
import sp_rl
from sp_rl.learning import SetCoverOracle
from sp_rl.agents import Graph
import time

def main(args):
  start_t = time.time()
  env = gym.make(args.env)
  env.seed(0)
  np.random.seed(0)
  _, graph_info = env.reset()
  #Parameters
  print('Creating oracle')
  
  total_reward = 0
  avg_time = 0.0

  for i in xrange(args.num_episodes):
    print "Episode = %d"%i
    G = Graph(graph_info['adj_mat'], graph_info['source_node'], graph_info['target_node'], pos=graph_info['pos'], edge_priors=graph_info['edge_priors'], lite_ftrs=True)
    obs, _ = env.reset()
    oracle = SetCoverOracle(env, G, args.horizon)#, k)
    value, time_taken = oracle.rollout_oracle(obs, render=args.render, step=args.step)
    total_reward += value
    print ('Episode reward = %d'%value)
    avg_time = avg_time + time_taken
  print('Number of episodes = {}, Total Reward = {}, Average reward = {}'.format(args.num_episodes, total_reward, total_reward*1.0/args.num_episodes*1.))

  # fig, ax = plt.subplots(2,1)
  # ax[0].plot(range(args.num_episodes), test_rewards)
  # ax[0].set_xlabel('Episodes')
  # ax[0].set_ylabel('Episode Reward')


  # ax[1].plot(range(args.num_episodes), test_avg_rewards)
  # ax[1].set_xlabel('Episodes')
  # ax[1].set_ylabel('Average Reward')
  avg_time = avg_time/args.num_episodes*1.0
  print "Average time taken = %f"%avg_time
  plt.show()



if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Policy that heuristically selects edges to expand')
  parser.add_argument('--env', type=str, required=True, help='environment name')
  parser.add_argument('--num_episodes', type=int, required=True)
  parser.add_argument('--render', action='store_true',
                      help='Render during testing')
  parser.add_argument("--step", action='store_true', help='If true, then agent pauses before executing each action to show rendering')
  parser.add_argument('--horizon', type=int, required=True, help='Lookahead horizon for set cover')


  args = parser.parse_args()
  main(args)
