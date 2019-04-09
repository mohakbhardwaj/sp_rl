#!/usr/bin/env python
import os,sys
sys.path.insert(0, os.path.abspath('..'))
import operator
import matplotlib.pyplot as plt
import numpy as np
import gym
import sp_rl
from sp_rl.agents import HeuristicAgent
import json
selectors = ['select_prior', 'select_posterior', 'select_lookahead_len', 'select_lookahead_prog', 'select_posterior_delta_len']
# selectors = ['length_oracle_2', 'select_lookahead_len']

def main(args):

  env = gym.make(args.env)
  env.seed(args.seed_val)
  np.random.seed(args.seed_val) 
  #Parameters
  for selector in selectors:
    print('Baseline = {}'.format(selector))
    # _, _ = env.reset(roll_back=True)
    ftr_params = None
    lite_ftrs = True
    agent = HeuristicAgent(env, selector, ftr_params, lite_ftrs)
    test_rewards_dict, test_avg_rewards_dict, avg_time = agent.test(args.num_episodes, render=args.render, step=args.step)

    test_rewards     = [it[1] for it in sorted(test_rewards_dict.items(), key=operator.itemgetter(0))]
    test_avg_rewards = [it[1] for it in sorted(test_avg_rewards_dict.items(), key=operator.itemgetter(0))]
  
    if not os.path.exists(args.folder):
      os.makedirs(args.folder)

    file_name = os.path.abspath(args.folder) + "/" + selector + "_" + str(args.num_episodes)
    results = dict()
    results['test_rewards'] = test_rewards_dict
    # results['test_avg_rewards'] = test_avg_rewards_dict
    print('Average Time = {}, Results dumped'.format(avg_time))
    with open(file_name +".json", 'w') as f:
      json.dump(results, f)
  



if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Policy that heuristically selects edges to expand')
  parser.add_argument('--env', type=str, required=True, help='environment name')
  parser.add_argument('--num_episodes', type=int, required=True)
  parser.add_argument('--render', action='store_true',
                      help='Render during testing')
  parser.add_argument("--step", action='store_true', help='If true, then agent pauses before executing each action to show rendering')
  parser.add_argument("--folder", type=str, required=True, help='Folder to store plots in')
  parser.add_argument("--seed_val", type=int, required=True, help='Seed value')
  args = parser.parse_args()
  main(args)
