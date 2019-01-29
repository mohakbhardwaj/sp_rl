#!/usr/bin/env python
"""Q learning with linear function approximation"""

try:
  import cPickle as pickle
except ImportError:  # python 3.x
  import pickle
import os,sys
sys.path.insert(0, os.path.abspath('..'))
import matplotlib.pyplot as plt
import numpy as np
import gym
import sp_rl
from sp_rl.agents import LinearQAgent, GraphWrapper


def main(args):
  env = gym.make(args.env)
  env.seed(0)
  np.random.seed(0)
  _, graph_info = env.reset()


  eps0 = 1.0
  epsf = 0.1
  gamma = 1.0
  alpha = 0.01
  k = 300 #Dataset 1
  # k = 350
  G = GraphWrapper(graph_info['adj_mat'], graph_info['source_node'], graph_info['target_node'], ftr_params=dict(k=k), pos=graph_info['pos'], edge_priors=graph_info['edge_priors'])
  w_init = np.ones((G.num_features, 1), dtype=np.float32)
  # w_init[3] = 100
  b_init = 0
  agent = LinearQAgent(env, eps0, epsf, gamma, alpha, w_init, b_init, G)

  if args.num_train_episodes is not None and args.num_exp_episodes is not None:
    rewards, avg_rewards, train_loss, train_avg_loss = agent.train(args.num_train_episodes, args.num_exp_episodes, render=False)
    weights = agent.q_fun.w
    bias = agent.q_fun.b
    print('(Training) Number of episodes = {}, Total Reward = {}, Average reward = {}'.format(args.num_train_episodes, sum(rewards), sum(rewards)*1.0/args.num_train_episodes*1.))
    print('Saving weights and params')
    
    data = dict()
    data['num_train_episodes'] = args.num_train_episodes
    data['num_exp_episodes'] = args.num_exp_episodes
    data['eps0'] = eps0
    data['gamma'] = gamma
    data['alpha'] = alpha
    data['weights'] = weights
    data['bias'] = bias
    with open(os.path.abspath(args.weight_file) + ".p", 'w') as fp:
      pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    fig1, ax1 = plt.subplots(2,2)
    ax1[0][0].plot(range(args.num_train_episodes), rewards)
    ax1[0][0].set_xlabel('Episodes')
    ax1[0][0].set_ylabel('Episode Reward')

    ax1[0][1].plot(range(args.num_train_episodes), avg_rewards)
    ax1[0][1].set_xlabel('Episodes')
    ax1[0][1].set_ylabel('Average Reward')
    
    ax1[1][0].plot(range(args.num_train_episodes), train_loss)
    ax1[1][0].set_xlabel('Episodes')
    ax1[1][0].set_ylabel('Episode Loss')

    ax1[1][1].plot(range(args.num_train_episodes), train_avg_loss)
    ax1[1][1].set_xlabel('Episodes')
    ax1[1][1].set_ylabel('Average Loss')
    fig1.suptitle('Train results', fontsize=16)
    plt.show(block=False)

  else:
    
    with open(os.path.abspath(args.weight_file) + ".p", 'r') as fp:
      data = pickle.load(fp)
      agent.q_fun.w = data['weights']
      agent.q_fun.b = data['bias']
      print data['weights'], data['bias']
      print data['num_train_episodes'], data['num_exp_episodes']
      raw_input('Weights loaded. Press enter to start testing...')

  test_rewards, test_avg_rewards, test_loss, test_avg_loss = agent.test(args.num_test_episodes, render=args.render, step=args.step)
  print('(Testing) Number of episodes = {}, Total Reward = {}, Average reward = {}'.format(args.num_test_episodes, sum(test_rewards), sum(test_rewards)*1.0/args.num_test_episodes*1.))

  
  fig2, ax2 = plt.subplots(2,2)
  ax2[0][0].plot(range(args.num_test_episodes), test_rewards)
  ax2[0][0].set_xlabel('Episodes')
  ax2[0][0].set_ylabel('Episode Reward')

  ax2[0][1].plot(range(args.num_test_episodes), test_avg_rewards)
  ax2[0][1].set_xlabel('Episodes')
  ax2[0][1].set_ylabel('Average Reward')

  ax2[1][0].plot(range(args.num_test_episodes), test_loss)
  ax2[1][0].set_xlabel('Episodes')
  ax2[1][0].set_ylabel('Episode Loss')

  ax2[1][1].plot(range(args.num_test_episodes), test_avg_loss)
  ax2[1][1].set_xlabel('Episodes')
  ax2[1][1].set_ylabel('Average Loss')



  fig2.suptitle('Test plots', fontsize=16)
  plt.show()


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Train tabular q learning on input environment')
  parser.add_argument('--env', type=str, required=True, help='environment name')
  parser.add_argument("--num_train_episodes", type=int, help='Number of training episodes')
  parser.add_argument("--num_exp_episodes", type=int, help='Number of pure exploration episodes before training')
  parser.add_argument("--num_test_episodes", type=int, help='Number of test episodes')
  parser.add_argument("--weight_file", required=True, type=str, help="File to store/load weights and parameters")
  parser.add_argument('--render', action='store_true', help='Render during testing')
  parser.add_argument("--step", action='store_true', help='If true, then agent pauses before executing each action to show rendering')
  
  args = parser.parse_args()
  main(args)


