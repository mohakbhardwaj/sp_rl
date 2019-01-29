#!/usr/bin/env python
"""Q learning with linear function approximation"""
import torch
import json
import os,sys
sys.path.insert(0, os.path.abspath('..'))
import matplotlib.pyplot as plt
import numpy as np
import gym
import sp_rl
from sp_rl.agents import AggrevateAgent, GraphWrapper
from sp_rl.learning import LinearNet, MLP, QFunction

torch.set_default_tensor_type(torch.DoubleTensor)
np.set_printoptions(threshold=np.nan, linewidth=np.inf)


def get_model_str(args, env):
  alg_name = "aggrevate"
  if args.use_advantage: alg_name = alg_name + "_advantage"
  model_str = alg_name + "_" + str(env.metadata['num_train_envs']) + "_" + str(env.metadata['num_validation_envs']) + "_" \
              + str(args.num_iters) + "_" + str(args.num_episodes_per_iter) \
              + args.model + "_" + args.expert
  if args.lite_ftrs: model_str  = model_str + "_lite_ftrs"
  model_str = model_str + "_" + str(args.seed_val)
  return model_str

def main(args):
  #Step 1: Create environments for training and validation and set their seeds
  env = gym.make(args.env)
  valid_env = gym.make(args.valid_env)
  env.seed(args.seed_val)
  np.random.seed(args.seed_val)
  valid_env.seed(args.seed_val)
  torch.manual_seed(args.seed_val)
  _, graph_info = env.reset()
  _, _ = valid_env.reset()
  
  #Step 2: Load learning parameters from provided experiment folder 
  with open(os.path.abspath(args.folder) + "/params.json", 'r') as fp:
    params = json.load(fp)

  beta0        = params['beta0']
  alpha        = params['alpha']
  k            = params['k']
  batch_size   = params['batch_size']
  train_epochs = params['train_epochs'] 
  weight_decay = params['weight_decay'] #L2 penalty
  T = params['T'] #Episode length for sampling oracle
  
  #Step 3: Create graph that the learner will use for path/feature calculation
  G = GraphWrapper(graph_info['adj_mat'], graph_info['source_node'], graph_info['target_node'], ftr_params=dict(k=k), pos=graph_info['pos'], edge_priors=graph_info['edge_priors'], lite_ftrs=args.lite_ftrs)
  #Step 4: Create function approximation model
  if args.model == 'linear': 
    model = LinearNet(G.num_features, 1)
    optim_method = 'sgd'
  elif args.model == 'mlp': 
    model = MLP(d_input=G.num_features, d_layers=[100, 50], d_output=1, activation=torch.nn.ReLU)
    optim_method = 'adam'
  
  print('Function approximation model = {}'.format(model))
  #Step 5: Create a learner that trains the model
  qfunc = QFunction(model, batch_size, train_method='supervised', optim_method=optim_method, lr=alpha, weight_decay=weight_decay, use_cuda=args.use_cuda)
  #Step 6: Finally create the agent
  agent = AggrevateAgent(env, valid_env, qfunc, beta0, T, args.expert, G, train_epochs)
  #Step 7: Train the agent and get results
  if args.num_iters is not None and args.num_valid_episodes is not None:
    rewards, avg_rewards, train_loss, train_avg_loss, validation_rewards, validation_avg_reward, data_points_per_iter = agent.train(args.num_iters, args.num_episodes_per_iter, args.num_valid_episodes, use_advantage=args.use_advantage)

    num_train_episodes = args.num_iters * args.num_episodes_per_iter
    print('(Training) Number of episodes = {}, Total Reward = {}, Average reward = {}, Best validation reward = {}'.format(num_train_episodes, sum(rewards), sum(rewards)*1.0/num_train_episodes*1., max(validation_rewards)))
    print('Saving weights and params')
    
    results = dict()
    results['train_rewards']          = rewards
    results['train_avg_rewards']      = avg_rewards
    results['train_loss']             = train_loss
    results['train_avg_loss']         = train_avg_loss
    results['validation_rewards']     = validation_rewards
    results['validation_avg_rewards'] = validation_avg_reward
    results['data_points_per_iter']   = data_points_per_iter
    
    print('Saving Results')
    with open(os.path.abspath(args.folder) + '/results.json', 'w') as fp:
      json.dump(results, fp)

    print('Saving model')    
    model_str = get_model_str(args, env)
    torch.save(agent.qfun.model.state_dict(), os.path.abspath(args.folder) + "/" + model_str)
    print('Model saved. Plot train results and run test')
    fig1, ax1 = plt.subplots(3,2)
    ax1[0][0].plot(range(num_train_episodes), rewards)
    ax1[0][0].set_xlabel('Episodes')
    ax1[0][0].set_ylabel('Episode Reward')

    ax1[0][1].plot(range(num_train_episodes), avg_rewards)
    ax1[0][1].set_xlabel('Episodes')
    ax1[0][1].set_ylabel('Average Reward')
    
    ax1[1][0].plot(range(args.num_iters), train_loss)
    ax1[1][0].set_xlabel('Iterations')
    ax1[1][0].set_ylabel('Regression Loss')

    ax1[1][1].plot(range(args.num_iters), train_avg_loss)
    ax1[1][1].set_xlabel('Iterations')
    ax1[1][1].set_ylabel('Regression Average Loss')
    fig1.suptitle('Train results', fontsize=16)

    
    ax1[2][0].plot(range(args.num_iters), validation_rewards)
    ax1[2][0].set_xlabel('Iterations')
    ax1[2][0].set_ylabel('Validation reward')

    ax1[2][1].plot(range(args.num_iters), validation_avg_reward)
    ax1[2][1].set_xlabel('Iterations')
    ax1[2][1].set_ylabel('Average valid reward')

    fig1.suptitle('Train results', fontsize=16)
    fig1.savefig(args.folder + 'train_results')
    plt.show(block=False)
    print('Running testing')
    test_rewards, test_avg_rewards = agent.test(agent.qfun, args.num_test_episodes, render=args.render, step=args.step)

  else:
    
    if not args.model_file:
      raise ValueError, 'Model file not specified'
    # agent.qfun.model.load_state_dict(torch.load(args.file_path))
    qfunc.model.load_state_dict(torch.load(os.path.join(args.folder, args.model_file)))
    if args.model == 'linear': print(qfunc.model.fc.weight, qfunc.model.fc.bias)
    # print('Loaded weights = {}'.format(qfunc.model.fc.weight))
    raw_input('Weights loaded. Press enter to start testing...')
    test_rewards, test_avg_rewards = agent.test(qfunc, args.num_test_episodes, render=args.render, step=args.step)
  print('(Testing) Number of episodes = {}, Total Reward = {}, Average reward = {}'.format(args.num_test_episodes, sum(test_rewards), sum(test_rewards)*1.0/args.num_test_episodes*1.))

  print('Plot test results and save')
  fig2, ax2 = plt.subplots(1,2)
  ax2[0].plot(range(args.num_test_episodes), test_rewards)
  ax2[0].set_xlabel('Episodes')
  ax2[0].set_ylabel('Episode Reward')

  ax2[1].plot(range(args.num_test_episodes), test_avg_rewards)
  ax2[1].set_xlabel('Episodes')
  ax2[1].set_ylabel('Average Reward')

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
  parser.add_argument("--model", type=str, help='Linear or MLP')
  parser.add_argument("--expert", type=str, help='Expert to eb used')
  parser.add_argument('--folder', required=True, type=str, help='Folder to load params.json and save results.json')
  parser.add_argument("--model_file", type=str, help="File to load model parameters during test phase")
  parser.add_argument("--use_advantage", action='store_true', help='Use advantage estimation from expert rather than just Q values')
  parser.add_argument('--render', action='store_true', help='Render during testing')
  parser.add_argument("--step", action='store_true', help='If true, then agent pauses before executing each action to show rendering')
  parser.add_argument("--lite_ftrs", action='store_true', help='Uses only bare minimum of features')
  parser.add_argument("--use_cuda", action='store_true', help='Use cuda for training')
  parser.add_argument("--seed_val", type=int, required=True, help='Seed value for training')
  args = parser.parse_args()
  main(args)


