#!/usr/bin/env python
"""Imitation learning using DAgger"""
import operator
import os,sys
sys.path.insert(0, os.path.abspath('..'))
import gym
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import sp_rl
from sp_rl.agents import CEMAgent, GraphWrapper
from sp_rl.learning import LinearNet, MLP, Policy

torch.set_default_tensor_type(torch.DoubleTensor)
np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)


def get_model_str(args, env):
  alg_name = "cem"
  model_str = alg_name + "_" \
              + str(args.num_iters) + "_" + str(args.num_episodes_per_iter) + "_" + str(args.num_valid_episodes) \
              + "_" + args.model + "_"
  model_str = model_str + "_" + str(args.seed_val) 
  return model_str

def main(args):
  env = gym.make(args.env)
  valid_env = gym.make(args.valid_env)
  env.seed(args.seed_val)
  np.random.seed(args.seed_val)
  valid_env.seed(args.seed_val)
  torch.manual_seed(args.seed_val)

  _, graph_info = env.reset()
  _, _ = valid_env.reset()
  
  
  G = GraphWrapper(graph_info)
  if args.quad_ftrs: n_inputs = 6 #15
  else: n_inputs = 5
  if args.model == 'linear': 
    model = LinearNet(n_inputs, 1)
    model.print_parameters()
    if args.model_file:
      model.load_state_dict(torch.load(os.path.join(args.folder, args.model_file)))
      model.print_parameters()
  # elif args.model == 'mlp': 
  #   model = MLP(d_input=n_inputs, d_layers=[n_inputs, n_inputs/3], d_output=1, activation=torch.nn.ReLU)
  
  agent  = CEMAgent(env, valid_env, model, G, args.batch_size, args.elite_frac, args.init_std)
  

  if not args.test:
    start_t = time.time()
    validation_rewards, weights_per_iter,= agent.train(args.num_iters, args.num_episodes_per_iter, 
                                                                                 args.num_valid_episodes, quad_ftrs=args.quad_ftrs)
    num_train_episodes = args.num_iters * args.num_episodes_per_iter
    # print('(Training) Number of episodes = {}, Total Reward = {}, Average reward = {}, \
    #         Best validation reward = {}, Train time = {}'.format(num_train_episodes, \
    #         sum(rewards), np.mean(rewards), max(validation_rewards), (time.time()-start_t)/60.0))
    # print('Saving weights and params')
    print('Training time = ', time.time() - start_t)
    results_dict = dict()
    results_dict['validation_rewards']  = validation_rewards

    if not os.path.exists(args.folder):
      os.makedirs(args.folder)
    
    with open(os.path.abspath(args.folder) + '/train_results.csv', 'w') as fp:
      writer = csv.writer(fp, dialect=csv.excel)
      for key, value in results_dict.items():
        row=[key]
        for v in value: row.append(v)
        writer.writerow(row)

    if args.model == "linear":
      print('Learned weights: ')
      agent.model.print_parameters()
      with open(os.path.abspath(args.folder) + '/weights_per_iter.json', 'w') as fp:
        json.dump(weights_per_iter, fp)
    model_str = get_model_str(args, env)
    torch.save(agent.model.state_dict(), os.path.abspath(args.folder) + "/" + model_str)


    if args.plot:
      print('Plot train results')
      fig1, ax1 = plt.subplots(2,1)
      ax1[0][0].plot(range(args.num_iters), np.median(train_rewards, axis=1))
      ax1[0][0].set_xlabel('CEM iteration')
      ax1[0][0].set_ylabel('Median Reward')
   
      if args.num_valid_episodes > 0:
        lquants = np.quantile(validation_rewards, 0.4, axis=1).reshape(1, args.num_iters)
        uquants = np.quantile(validation_rewards, 0.6, axis=1).reshape(1, args.num_iters)
        yerr=np.concatenate((lquants, uquants), axis=0)
        #ax1[2][0].errorbar(range(args.num_iters), np.median(validation_rewards,axis=1), yerr=yerr)
        ax1[1][0].plot(range(args.num_iters), np.median(validation_rewards,axis=1))
        ax1[1][0].set_xlabel('CEM iteration')
        ax1[1][0].set_ylabel('Validation median reward')

      fig1.suptitle('Train results', fontsize=16)
      fig1.savefig(args.folder + '/train_results.pdf', bbox_inches='tight')
      plt.show(block=False)

    print('Running final testing')
    _,_ = valid_env.reset(roll_back=True)
    test_rewards_dict = agent.test(valid_env, agent.model, args.num_test_episodes, render=args.render, step=args.step, quad_ftrs=args.quad_ftrs)
  else:
    if not args.model_file:
      raise ValueError, 'Model file not specified'
    model.load_state_dict(torch.load(os.path.join(args.folder, args.model_file)))
    if args.model == 'linear': print(model.fc.weight, model.fc.bias)
    test_env = gym.make(args.valid_env)
    test_env.seed(args.seed_val)
    _, _ = test_env.reset()
    # raw_input('Weights loaded. Press enter to start testing...')
    test_rewards_dict = agent.test(test_env, model, args.num_test_episodes, render=args.render, step=args.step, quad_ftrs=args.quad_ftrs)
  

  test_rewards = [it[1] for it in sorted(test_rewards_dict.items(), key=operator.itemgetter(0))]
  test_results = dict()
  test_results['test_rewards'] = test_rewards_dict
  print ('Average test rewards = {}'.format(np.mean(test_rewards)))
  print('Dumping results')
  with open(os.path.abspath(args.folder) + '/test_results.json', 'w') as fp:
    json.dump(test_results, fp)

  print('(Testing) Number of episodes = {}, Total Reward = {}, Average reward = {},\
          Std. Dev = {}, Median reward = {}'.format(args.num_test_episodes, sum(test_rewards), \
          np.mean(test_rewards), np.std(test_rewards), np.median(test_rewards)))

  if args.plot:
    print('Plot test results and save')
    fig2, ax2 = plt.subplots(1,1)
    ax2[0].plot(range(len(test_rewards)), test_rewards)
    ax2[0].set_xlabel('Episodes')
    ax2[0].set_ylabel('Episode Reward')

    fig2.suptitle('Test plots', fontsize=16)
    fig2.savefig(args.folder+"test_results")
    plt.show()


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Train tabular q learning on input environment')
  parser.add_argument('--env', type=str, required=True, help='environment name')
  parser.add_argument('--valid_env', type=str, required=True, help='environment name for validation/testing')
  parser.add_argument("--num_iters", type=int, help='Number of training episodes')
  parser.add_argument("--num_episodes_per_iter", type=int, help='Number of episodes per iteration')
  parser.add_argument("--num_valid_episodes", type=int, help='Number of validation episodes')
  parser.add_argument("--num_test_episodes", type=int, help='Number of test episodes')
  parser.add_argument("--batch_size", type=int)
  parser.add_argument("--elite_frac", type=float, default=0.2)
  parser.add_argument("--init_std", type=float, default=1.0)
  parser.add_argument('--folder', required=True, type=str, help='Folder to load params.json and save results.json')
  parser.add_argument("--model", type=str, help='Linear or MLP')
  parser.add_argument("--model_file", type=str, help="File to load model initial parameters or parameters for testing")
  parser.add_argument('--render', action='store_true', help='Render during testing')
  parser.add_argument("--step", action='store_true', help='If true, then agent pauses before executing each action to show rendering')
  parser.add_argument("--seed_val", type=int, required=True, help='Seed value for training')
  parser.add_argument("--test", action='store_true', help='Loads weights from the model file and runs testing')
  parser.add_argument("--plot", action='store_true', help='Whether to plot results or not')
  parser.add_argument("--quad_ftrs", action='store_true', help="Use quadratic features or not")
  args = parser.parse_args()
  main(args)


