#!/usr/bin/env python
import os,sys
sys.path.insert(0, os.path.abspath('..'))
import operator
# import matplotlib
# matplotlib.use('cairo')
import matplotlib.pyplot as plt
import numpy as np
import gym
import sp_rl
from sp_rl.agents import HeuristicAgent
import json
import csv


def main(args):
  env = gym.make(args.env)
  env.seed(args.seed_val)
  np.random.seed(args.seed_val) 
  for selector in args.heuristics:
    print('Heuristic selector = {}'.format(selector))
    agent = HeuristicAgent(env, selector, args.k)
    test_rewards_dict, test_edge_ids_dict, avg_time = agent.test(args.num_episodes, render=args.render, step=args.step)
    test_rewards     = [it[1] for it in sorted(test_rewards_dict.items(), key=operator.itemgetter(0))]
    print test_rewards
    if not os.path.exists(args.folder):
      os.makedirs(args.folder)
    file_name = os.path.abspath(args.folder) + "/" + selector + "_" + str(args.num_episodes)    
    # if args.k is not None:
      # file_name = file_name + "_" + str(args.k)
    # file_name = file_name + "_" + str(args.seed_val)
    results = dict()
    results['test_rewards']  = test_rewards_dict
    with open(file_name +".json", 'w') as f:
      json.dump(results, f)
    with open(file_name + '_edge_ids.csv', 'w') as fp:
      writer = csv.writer(fp, dialect=csv.excel)
      for key, value in test_edge_ids_dict.items():
        row=[key]
        for v in value: row.append(v)
        writer.writerow(row)



if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Policy that heuristically selects edges to expand')
  parser.add_argument('--env', type=str, required=True, help='environment name')
  parser.add_argument('--heuristics', type=str, nargs='+', required=True)
  parser.add_argument('--num_episodes', type=int, required=True)
  parser.add_argument('--k', type=int)
  parser.add_argument('--render', action='store_true', help='Render during testing')
  parser.add_argument("--step", action='store_true', help='If true, then agent pauses before executing each action to show rendering')
  parser.add_argument("--folder", type=str, required=True, help='Folder to store plots in')
  parser.add_argument("--seed_val", type=int, required=True, help='Seed value')
  args = parser.parse_args()
  main(args)
