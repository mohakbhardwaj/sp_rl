#!/usr/bin/env python
import os,sys
sys.path.insert(0, os.path.abspath('..'))
import operator
import matplotlib.pyplot as plt
import numpy as np
import gym
import sp_rl
from sp_rl.agents import HeuristicAgent, Graph
import json

def main(args):
  env = gym.make(args.env)
  env.seed(args.seed_val)
  np.random.seed(args.seed_val)

  #Parameters
  selector = args.heuristic
  ftr_params = None
  lite_ftrs = False
  if selector in ["select_k_shortest", "select_priorkshortest", "select_lookahead"]:
    ftr_params = dict(k=args.k)
    lite_ftrs = False

  agent = HeuristicAgent(env, selector, ftr_params, lite_ftrs)
  test_rewards_dict, test_avg_rewards_dict, avg_time = agent.test(args.num_episodes, render=args.render, step=args.step)

  test_rewards     = [it[1] for it in sorted(test_rewards_dict.items(), key=operator.itemgetter(0))]
  test_avg_rewards = [it[1] for it in sorted(test_avg_rewards_dict.items(), key=operator.itemgetter(0))]
  # print len(test_rewards)
  # print test_rewards
  # print test_avg_rewards
  # print('Number of episodes = {}, Total Reward = {}, Average reward = {}, Average time = {}'.format(args.num_episodes, sum(test_rewards), np.mean(test_rewards), avg_time))

  # fig, ax = plt.subplots(2,1)
  # ax[0].plot(range(len(test_rewards)), test_rewards)
  # ax[0].set_xlabel('Episodes')
  # ax[0].set_ylabel('Episode Reward')


  # ax[1].plot(range(len(test_avg_rewards)), test_avg_rewards)
  # ax[1].set_xlabel('Episodes')
  # ax[1].set_ylabel('Average Reward')
  
  if not os.path.exists(args.folder):
    os.makedirs(args.folder)

  file_name = os.path.abspath(args.folder) + "/" + args.heuristic + "_" + str(args.num_episodes)
  if args.k is not None:
    file_name = file_name + "_" + str(args.k)
  file_name = file_name + "_" + str(args.seed_val)
  results = dict()
  results['test_rewards'] = test_rewards_dict
  results['test_avg_rewards'] = test_avg_rewards_dict
  print('Average Time = {}, Results dumped'.format(avg_time))
  with open(file_name +".json", 'w') as f:
    json.dump(results, f)
  # fig.savefig(file_name)
  
  # plt.show(block=False)



if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Policy that heuristically selects edges to expand')
  parser.add_argument('--env', type=str, required=True, help='environment name')
  parser.add_argument('--heuristic', type=str, required=True)
  parser.add_argument('--num_episodes', type=int, required=True)
  parser.add_argument('--k', type=int)
  parser.add_argument('--render', action='store_true',
                      help='Render during testing')
  parser.add_argument("--step", action='store_true', help='If true, then agent pauses before executing each action to show rendering')
  parser.add_argument("--folder", type=str, required=True, help='Folder to store plots in')
  parser.add_argument("--seed_val", type=int, required=True, help='Seed value')
  args = parser.parse_args()
  main(args)
