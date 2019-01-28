#!/usr/bin/env python
import os,sys
sys.path.insert(0, os.path.abspath('..'))
import matplotlib.pyplot as plt
import numpy as np
import gym
import sp_rl
from sp_rl.agents import QTableAgent


def main(args):
  env = gym.make(args.env)
  env.seed(0)
  np.random.seed(0)

  #Parameters
  # num_episodes = 3500 #Number of training episodes
  # num_exp_episodes = 150 #Number of episodes of pure random exploration
  # num_test_episodes = 1000
  eps0 = 1.0
  gamma = 1.0
  alpha = 0.5
  agent = QTableAgent(env, eps0, gamma, alpha)

  rewards, avg_rewards = agent.train(args.num_train_episodes, args.num_exp_episodes, render=False)
  test_rewards, test_avg_rewards = agent.test(args.num_test_episodes, render=args.render, step=args.step)

  print('Number of episodes = {}, Total Reward = {}, Average reward = {}'.format(args.num_train_episodes, sum(rewards), sum(rewards)*1.0/args.num_train_episodes*1.))

  fig1, ax1 = plt.subplots(2,1)
  ax1[0].plot(range(args.num_train_episodes), rewards)
  ax1[0].set_xlabel('Episodes')
  ax1[0].set_ylabel('Episode Reward')
  ax1[0].set_ylim(-10, 0)

  ax1[1].plot(range(args.num_train_episodes), avg_rewards)
  ax1[1].set_xlabel('Episodes')
  ax1[1].set_ylabel('Average Reward')
  fig1.suptitle('Train results', fontsize=16)
  
  fig2, ax2 = plt.subplots(2,1)
  ax2[0].plot(range(args.num_test_episodes), test_rewards)
  ax2[0].set_xlabel('Episodes')
  ax2[0].set_ylabel('Episode Reward')
  ax2[0].set_ylim(-10, 0)

  ax2[1].plot(range(args.num_test_episodes), test_avg_rewards)
  ax2[1].set_xlabel('Episodes')
  ax2[1].set_ylabel('Average Reward')
  fig2.suptitle('Test plots', fontsize=16)
  plt.show()




if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Train tabular q learning on input environment')
  parser.add_argument('--env', type=str, required=True, help='environment name')
  parser.add_argument("--num_train_episodes", type=int, required=True, help='Number of training episodes')
  parser.add_argument("--num_exp_episodes", type=int, required=True, help='Number of pure exploration episodes before training')
  parser.add_argument("--num_test_episodes", type=int, required=True, help='Number of test episodes')
  parser.add_argument('--render', action='store_true', help='Render during testing')
  parser.add_argument("--step", action='store_true', help='If true, then agent pauses before executing each action to show rendering')
  args = parser.parse_args()
  main(args)


