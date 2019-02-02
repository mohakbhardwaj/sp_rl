#!/usr/bin/env python
"""Imitation learning using DAgger"""
import operator
import os,sys
sys.path.insert(0, os.path.abspath('..'))
import gym
import json
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import sp_rl
from sp_rl.agents import QLearningAgent, GraphWrapper
from sp_rl.learning import LinearNet, MLP, QFunction

torch.set_default_tensor_type(torch.DoubleTensor)
# np.set_printoptions(threshold=np.nan, linewidth=np.inf)


def get_model_str(args, env):
  alg_name = "qlearn"
  model_str = alg_name + "_" \
              + str(args.num_episodes) + "_" + str(args.num_exp_episodes) \
              + "_" + args.model
  model_str = model_str + "_" + str(args.eps0) + "_" + str(args.alpha)  \
                        + "_" + str(args.batch_size) + "_" + str(args.weight_decay)
  if args.lite_ftrs: model_str  = model_str + "_lite_ftrs"
  else: model_str = model_str + "_" + str(args.k)
  model_str = model_str + "_" + str(args.seed_val) 
  return model_str

def main(args):
  env = gym.make(args.env)
  env.seed(args.seed_val)
  np.random.seed(args.seed_val)
  torch.manual_seed(args.seed_val)

  _, graph_info = env.reset(roll_back=True)
  

  eps0             = args.eps0
  epsf             = args.epsf
  alpha            = args.alpha
  k                = args.k
  batch_size       = args.batch_size
  train_epochs     = args.epochs 
  weight_decay     = args.weight_decay #L2 penalty
  
  G = GraphWrapper(graph_info['adj_mat'], graph_info['source_node'], graph_info['target_node'], ftr_params=dict(k=k), pos=graph_info['pos'], edge_priors=graph_info['edge_priors'], lite_ftrs=args.lite_ftrs)
  if args.model == 'linear': 
    model = LinearNet(G.num_features, 1)
    optim_method = 'sgd'
  elif args.model == 'mlp': 
    model = MLP(d_input=G.num_features, d_layers=[100, 50], d_output=1, activation=torch.nn.ReLU)
    optim_method = 'adam'
  # print('Function approximation model = {}'.format(model))
  qfun = QFunction(model, batch_size, train_method='reinforcement', optim_method=optim_method, lr=alpha, weight_decay=weight_decay, use_cuda=args.use_cuda)
  agent = QLearningAgent(env, qfun, G, eps0, epsf, batch_size)
  
  # if not args.test:
  start_t = time.time()
  # print args.num_iters, args.num_episodes_per_iter
  train_rewards = agent.train(args.num_episodes, args.num_exp_episodes)
  # print('(Training) Number of episodes = {}, Total Reward = {}, Average reward = {}, \
  #         Best validation reward = {}, Train time = {}'.format(num_train_episodes, \
  #         sum(rewards), np.mean(rewards), max(validation_rewards), (time.time()-start_t)/60.0))
  # print('Saving weights and params')
  
  results = dict()
  results['train_rewards']          = train_rewards
  # results['train_avg_rewards']      = avg_rewards
  # results['train_loss']             = train_loss
  # results['train_avg_loss']         = train_avg_loss
  # results['validation_rewards']     = validation_rewards
  # results['validation_stds']        = validation_stds
  # results['validation_avg_rewards'] = validation_avg_reward
  # results['data_points_per_iter']   = data_points_per_iter
  # if args.model == "linear":
    # results['state_dict_per_iter']    = state_dict_per_iter
  
  # print('Saving Results')
  if not os.path.exists(args.folder):
    os.makedirs(args.folder)
  
  with open(os.path.abspath(args.folder) + '/train_results.json', 'w') as fp:
    json.dump(results, fp)
      
  model_str = get_model_str(args, env)
  torch.save(agent.q_fun.model.state_dict(), os.path.abspath(args.folder) + "/" + model_str)
  
  # for s in state_dict_per_iter:
  #   st_dct = state_dict_per_iter[s]
  #   file_name = model_str + "_" + 'policy_int_' + str(s)
  #   torch.save(st_dct, os.path.abspath(args.folder)  + "/" + file_name)
  
  # for it in features_per_iter:
  #   file_name_f = "features_" + str(it)
  #   file_name_l = "labels_" + str(it)
  #   with open(os.path.abspath(args.folder) + '/' + file_name_f, 'w') as fp:
  #     json.dump(features_per_iter[it], fp)
  #   with open(os.path.abspath(args.folder) + '/' + file_name_l, 'w') as fp:
  #     json.dump(labels_per_iter[it], fp)


  if args.plot:
    print('Plot train results')
    fig1, ax1 = plt.subplots(3,2)
    ax1[0][0].plot(range(num_train_episodes), train_rewards)
    ax1[0][0].set_xlabel('Episodes')
    ax1[0][0].set_ylabel('Episode Reward')

      # ax1[0][1].plot(range(num_train_episodes), avg_rewards)
      # ax1[0][1].set_xlabel('Episodes')
      # ax1[0][1].set_ylabel('Average Reward')
      
      # ax1[1][0].plot(range(args.num_iters), train_loss)
      # ax1[1][0].set_xlabel('Iterations')
      # ax1[1][0].set_ylabel('Regression Loss')

      # ax1[1][1].plot(range(args.num_iters), train_avg_loss)
      # ax1[1][1].set_xlabel('Iterations')
      # ax1[1][1].set_ylabel('Regression Average Loss')
      # fig1.suptitle('Train results', fontsize=16)

      
      # ax1[2][0].errorbar(range(args.num_iters), validation_rewards, yerr=validation_stds)
      # ax1[2][0].set_xlabel('Iterations')
      # ax1[2][0].set_ylabel('Validation reward')

      # ax1[2][1].plot(range(args.num_iters), validation_avg_reward)
      # ax1[2][1].set_xlabel('Iterations')
      # ax1[2][1].set_ylabel('Average valid reward')

      # fig1.suptitle('Train results', fontsize=16)
      # fig1.savefig(args.folder + 'train_results')
      # plt.show(block=False)

    # test_env = gym.make(args.valid_env)
    # test_env.seed(args.seed_val)
    # _, _ = test_env.reset()
    # _,_ = valid_env.reset(roll_back=True)
    # if model == "linear":
    #   print('Learned weights = {}'.format(agent.policy.model.state_dict()))
    # test_rewards_dict, test_avg_rewards_dict, test_acc_dict = agent.test(valid_env, agent.policy, args.num_test_episodes, render=args.render, step=args.step)

  # else:
  #   print args.test
  #   if not args.model_file:
  #     raise ValueError, 'Model file not specified'
  #   policy.model.load_state_dict(torch.load(os.path.join(args.folder, args.model_file)))
  #   if args.model == 'linear': print(policy.model.fc.weight, policy.model.fc.bias)
  #   test_env = gym.make(args.valid_env)
  #   test_env.seed(args.seed_val)
  #   _, _ = test_env.reset()
  #   # raw_input('Weights loaded. Press enter to start testing...')
  #   test_rewards_dict, test_avg_rewards_dict, test_acc_dict = agent.test(test_env, policy, args.num_test_episodes, render=args.render, step=args.step)
  
  # test_rewards     = [it[1] for it in sorted(test_rewards_dict.items(), key=operator.itemgetter(0))]
  # test_avg_rewards = [it[1] for it in sorted(test_avg_rewards_dict.items(), key=operator.itemgetter(0))]
  # print test_rewards
  # test_results = dict()
  # test_results['test_rewards'] = test_rewards_dict
  # test_results['test_acc'] = test_acc_dict
  # print ('Average test rewards = {}'.format(np.mean(test_rewards)))
  # print('Dumping results')
  # # with open(os.path.abspath(args.folder) + '/test_results.json', 'w') as fp:
  # #   json.dump(test_results, fp)

  # # print('(Testing) Number of episodes = {}, Total Reward = {}, Average reward = {}, \
  # #         Std. Dev = {}'.format(args.num_test_episodes, sum(test_rewards), \
  # #         np.mean(test_rewards), np.std(test_rewards)))

  # if args.plot:
  #   print('Plot test results and save')
  #   fig2, ax2 = plt.subplots(1,2)
  #   ax2[0].plot(range(len(test_rewards)), test_rewards)
  #   ax2[0].set_xlabel('Episodes')
  #   ax2[0].set_ylabel('Episode Reward')

  #   ax2[1].plot(range(len(test_avg_rewards)), test_avg_rewards)
  #   ax2[1].set_xlabel('Episodes')
  #   ax2[1].set_ylabel('Average Reward')

  #   fig2.suptitle('Test plots', fontsize=16)
  #   fig2.savefig(args.folder+"test_results")
  #   plt.show()


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Train tabular q learning on input environment')
  parser.add_argument('--env', type=str, required=True, help='environment name')
  parser.add_argument("--num_episodes", type=int, help='Number of test episodes')
  parser.add_argument("--num_exp_episodes", type=int, help='Number of test episodes')
  parser.add_argument("--model", type=str, help='Linear or MLP')
  parser.add_argument('--folder', required=True, type=str, help='Folder to load params.json and save results.json')
  parser.add_argument('--eps0', type=float, default=1.0, help='Initial epsilon')
  parser.add_argument('--epsf', type=float, default=0.1, help='Final epsilon')
  parser.add_argument('--alpha', type=float, default=0.01, help='Initial learning rate')
  parser.add_argument('--k'    , type=int, default=1, help='Feature parameter - Number of future paths to consider (only used if lite_ftrs is off)')
  parser.add_argument('--batch_size', type=int, default=32, help='Batch size for SGD')
  parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for SGD')
  parser.add_argument('--weight_decay', type=float, default=0.2, help='L2 Regularization')
  parser.add_argument("--model_file", type=str, help="File to load model parameters during test phase")
  parser.add_argument('--render', action='store_true', help='Render during testing')
  parser.add_argument("--step", action='store_true', help='If true, then agent pauses before executing each action to show rendering')
  parser.add_argument("--lite_ftrs", action='store_true', help='Uses only bare minimum of features')
  parser.add_argument("--use_cuda", action='store_true', help='Use cuda for training')
  parser.add_argument("--seed_val", type=int, required=True, help='Seed value for training')
  parser.add_argument("--re_init", action='store_true', help='Whether to re-initialize policy at every iteration of DAgger training')
  parser.add_argument("--test", action='store_true', help='Loads weights from the model file and runs testing')
  parser.add_argument("--plot", action='store_true', help='Whether to plot results or not')
  args = parser.parse_args()
  main(args)


