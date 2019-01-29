#!/usr/bin/env python
import os,sys
sys.path.insert(0, os.path.abspath('..'))
import operator
import matplotlib.pyplot as plt
import numpy as np
import gym
import sp_rl
from sp_rl.agents import DaggerAgent, GraphWrapper
from sp_rl.learning import LinearNet, MLP, Policy
import json
from copy import deepcopy
import torch
import time
torch.set_default_tensor_type(torch.DoubleTensor)



def get_model_str(args, env):
  alg_name = "dagger"
  model_str = alg_name + "_" \
              + str(args.num_iters) + "_" + str(args.num_episodes_per_iter) + "_" + str(args.num_valid_episodes) \
              + "_" + args.model + "_" + args.expert
  model_str = model_str + "_" + str(args.beta0) + "_" + str(args.alpha)  \
                        + "_" + str(args.batch_size) + "_" + str(args.epochs) + "_" + str(args.weight_decay)
  if args.lite_ftrs: model_str  = model_str + "_lite_ftrs"
  else: model_str = model_str + "_" + str(args.k)
  model_str = model_str + "_" + str(args.seed_val) 
  return model_str


def filter_path(path, obs):
  """This function filters out edges that have already been evaluated"""
  path_f = list(filter(lambda x: obs[x] == -1, path))
  return path_f

def get_path(graph, env):
  sp = graph.curr_shortest_path
  sp = env.to_edge_path(sp) 
  return sp

def render_env(env, path, scores):
  edge_widths={}
  edge_colors={}
  for k,i in enumerate(path):
    edge_widths[i] = 5.0
    color_val = 1.0 - (scores[k][0] - np.min(scores))/(np.max(scores) - np.min(scores))
    color_val = max(min(color_val, 0.65), 0.1)
    edge_colors[i] = str(color_val)
    # print color_val
  env.render(edge_widths=edge_widths, edge_colors=edge_colors)

def test(env, G, policy, num_episodes, render=False, step=False):
  test_rewards = {}
  _, _ = env.reset(roll_back=True)
  policy.model.eval() #Put the model in eval mode to turn off dropout etc.
  policy.model.cpu() #Copy over the model to cpu if on cuda 
  start_t = time.time()
  with torch.no_grad(): #Turn off autograd
    for i in range(num_episodes):
      j = 0
      ep_reward = 0
      obs, _ = env.reset()
      G.reset()
      path = get_path(G, env)
      ftrs = torch.tensor(G.get_features([env.edge_from_action(a) for a in path], j))
      probs = policy.predict(ftrs)
      probs = probs.detach().numpy()
      if render:
        render_env(env, path, probs)
      done = False
      # print('Curr episode = {}'.format(i+1))
      while not done:
        path = get_path(G, env)
        feas_actions = filter_path(path, obs)
        ftrs = torch.tensor(G.get_features([env.edge_from_action(a) for a in feas_actions], j))
        probs = policy.predict(ftrs)
        probs = probs.detach().numpy()
        if render:
          render_env(env, feas_actions, probs)
        #select greedy action
        idx = np.random.choice(np.flatnonzero(probs==probs.max())) #random tie breaking
        act_id = feas_actions[idx]
        act_e = env.edge_from_action(act_id)
        # print q_vals[idx], ftrs[idx]
        # print ('Q_vals = {}, feasible_actions = {}, chosen edge = {}, features = {}'.format(q_vals, feas_actions, act_e, ftrs))
        if step: raw_input('Press enter to execute action')
        obs, reward, done, info = env.step(act_id)
        # print ('obs = {}, reward = {}, done = {}'.format(obs, reward, done))    
        G.update_edge(act_e, obs[act_id])#Update the edge weight according to the result
                
        ep_reward += reward
        j += 1
      
      if render:
        fr = torch.tensor(G.get_features([env.edge_from_action(a) for a in path], j))
        sc = policy.predict(fr)
        render_env(env, path, sc.detach().numpy())
        raw_input('Press Enter')
          

      test_rewards[env.world_num] = ep_reward
      if step: raw_input('Episode over, press enter for next episode')

  avg_time = (time.time() - start_t)*1.0/num_episodes*1.0
  return test_rewards, avg_time


def main(args):
  # print('Function approximation model = {}'.format(model))
  env = gym.make(args.env)
  env.seed(args.seed_val)
  np.random.seed(args.seed_val)
  torch.manual_seed(args.seed_val) 
  #Parameters
  _, graph_info = env.reset(roll_back=True)

  G = GraphWrapper(graph_info['adj_mat'], graph_info['source_node'], graph_info['target_node'], ftr_params=None, pos=graph_info['pos'], edge_priors=graph_info['edge_priors'], lite_ftrs=args.lite_ftrs)
  if args.model == 'linear': 
    model = LinearNet(G.num_features, 1)
    optim_method = 'sgd'
  elif args.model == 'mlp': 
    model = MLP(d_input=G.num_features, d_layers=[30, 20], d_output=1, activation=torch.nn.ReLU)
    optim_method = 'adam'
  policy = Policy(model, 32, train_method='supervised', optim_method=optim_method, lr=0.01, weight_decay=0.2, use_cuda=False)

  best_reward = -np.inf
  best_test_rewards_dict = {}
  best_policy=  None
  best_iter = -1
  
  med_rewards=[]
  print(args.files)
  for i,file in enumerate(args.files):
    print('Policy number = {}, best reward = {} at iteration = {}'.format(i, best_reward, best_iter))
    _, _ = env.reset(roll_back=True)
    G.reset()
    policy.model.load_state_dict(torch.load(os.path.abspath(args.folder)+"/"+file))
    if args.model == 'linear': print(policy.model.fc.weight, policy.model.fc.bias)
    test_rewards_dict, avg_time = test(env, G, policy, args.num_episodes, render=args.render, step=args.step)
    test_rewards     = [it[1] for it in sorted(test_rewards_dict.items(), key=operator.itemgetter(0))]

    median_reward = np.median(test_rewards)
    print('Current median reward = {}'.format(median_reward))
    med_rewards.append(median_reward)
    if median_reward >= best_reward:
      print('Best so far...saving policy number = {}'.format(i))
      best_reward = median_reward
      best_policy = deepcopy(policy)
      best_iter = i
      best_test_rewards_dict = test_rewards_dict

  if not os.path.exists(args.folder):
    os.makedirs(args.folder)

  file_name = os.path.abspath(args.folder) + "/" + 'best_policy_test_rewards' + "_" + str(args.num_episodes)
  results = dict()
  results['test_rewards_best'] = best_test_rewards_dict
  print('Average Time = {}, Results dumped'.format(avg_time))
  with open(file_name +".json", 'w') as f:
    json.dump(results, f)
  print('Saving best policy')
  torch.save(best_policy.model.state_dict(), os.path.abspath(args.folder) + "/" + 'best_policy')
  plt.plot(med_rewards)
  plt.show()



if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Policy that heuristically selects edges to expand')
  parser.add_argument('--env', type=str, required=True, help='environment name')
  parser.add_argument('--files', type=str, required=True, nargs='+', help='Files with weights')
  parser.add_argument('--num_episodes', type=int, required=True)
  parser.add_argument('--render', action='store_true', help='Render during testing')
  parser.add_argument("--step", action='store_true', help='If true, then agent pauses before executing each action to show rendering')
  parser.add_argument("--folder", type=str, required=True, help='Folder to store final results in')
  parser.add_argument("--seed_val", type=int, required=True, help='Seed value')
  parser.add_argument("--lite_ftrs", action='store_true', help='whether to use lite ftrs or not')
  parser.add_argument("--model", type=str, required='True')
  args = parser.parse_args()
  main(args)
