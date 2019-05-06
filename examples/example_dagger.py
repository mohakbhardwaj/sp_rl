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
from sp_rl.agents import DaggerAgent, GraphWrapper
from sp_rl.learning import LinearNet, MLP, Policy

torch.set_default_tensor_type(torch.DoubleTensor)
np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)


def get_model_str(args, env):
  alg_name = "dagger"
  model_str = alg_name + "_" \
              + str(args.num_iters) + "_" + str(args.num_episodes_per_iter) + "_" + str(args.num_valid_episodes) \
              + "_" + args.model + "_" + args.expert
  model_str = model_str + "_" + str(args.beta0) + "_" + str(args.alpha)  \
                        + "_" + str(args.batch_size) + "_" + str(args.epochs) + "_" + str(args.weight_decay)
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
  

  beta0        = args.beta0
  alpha        = args.alpha
  momentum     = args.momentum
  gamma        = args.gamma
  batch_size   = args.batch_size
  train_epochs = args.epochs 
  weight_decay = args.weight_decay #L2 penalty
  
  G = GraphWrapper(graph_info)
  if args.quad_ftrs: n_inputs = 15
  else: n_inputs = 5
  if args.model == 'linear': 
    model = LinearNet(n_inputs, 1)
    optim_method = 'sgd'
  elif args.model == 'mlp': 
    model = MLP(d_input=n_inputs, d_layers=[n_inputs, n_inputs/3], d_output=1, activation=torch.nn.ReLU)
    optim_method = 'adam'
  # print('Function approximation model = {}'.format(model))
  policy = Policy(model, batch_size, train_method='supervised', optim_method=optim_method, lr=alpha, momentum = momentum, weight_decay=weight_decay, use_cuda=args.use_cuda)
  agent  = DaggerAgent(env, valid_env, policy, beta0, gamma, args.expert, G, train_epochs)
  
  if not args.test:
    start_t = time.time()
    train_rewards, train_loss, train_accs, validation_reward, \
    validation_accuracy, dataset_size, weights_per_iter, \
    features_per_iter, labels_per_iter = agent.train(args.num_iters, args.num_episodes_per_iter, args.num_valid_episodes,\
                                                     heuristic=args.heuristic, re_init=args.re_init, mixed_rollin=args.mixed_rollin, quad_ftrs=args.quad_ftrs)
    num_train_episodes = args.num_iters * args.num_episodes_per_iter
    # print('(Training) Number of episodes = {}, Total Reward = {}, Average reward = {}, \
    #         Best validation reward = {}, Train time = {}'.format(num_train_episodes, \
    #         sum(rewards), np.mean(rewards), max(validation_rewards), (time.time()-start_t)/60.0))
    # print('Saving weights and params')
    print('Training time = ', time.time() - start_t)
    results_dict = dict()
    results_dict['train_rewards']       = train_rewards
    results_dict['train_loss']          = train_loss
    results_dict['train_accs']          = train_accs
    results_dict['validation_rewards']  = validation_reward
    results_dict['validation_accuracy'] = validation_accuracy
    results_dict['dataset_size']        = dataset_size

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
      agent.policy.model.print_parameters()
    model_str = get_model_str(args, env)
    torch.save(agent.policy.model.state_dict(), os.path.abspath(args.folder) + "/" + model_str)

    if args.model == 'linear':
      with open(os.path.abspath(args.folder) + '/weights_per_iter.json', 'w') as fp:
        json.dump(weights_per_iter, fp)


    
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
      ax1[0][0].plot(range(args.num_iters), np.median(train_rewards, axis=1))
      ax1[0][0].set_xlabel('DAgger iteration')
      ax1[0][0].set_ylabel('Median Reward')

      ax1[0][1].plot(range(args.num_iters), np.mean(train_accs, axis=1))
      ax1[0][1].set_xlabel('DAgger iteration')
      ax1[0][1].set_ylabel('Avg. accuracy')
      
      ax1[1][0].plot(range(args.num_iters), train_loss)
      ax1[1][0].set_xlabel('DAgger iteration')
      ax1[1][0].set_ylabel('Regression Loss')
   
      if args.num_valid_episodes > 0:
        lquants = np.quantile(validation_reward, 0.4, axis=1).reshape(1, args.num_iters)
        uquants = np.quantile(validation_reward, 0.6, axis=1).reshape(1, args.num_iters)
        yerr=np.concatenate((lquants, uquants), axis=0)
        ax1[2][0].errorbar(range(args.num_iters), np.median(validation_reward,axis=1), yerr=yerr)
        ax1[2][0].set_xlabel('Iterations')
        ax1[2][0].set_ylabel('Validation reward')

        ax1[2][1].plot(range(args.num_iters), np.mean(validation_accuracy, axis=1))
        ax1[2][1].set_xlabel('DAgger iteration')
        ax1[2][1].set_ylabel('Validation average accuracy')

      fig1.suptitle('Train results', fontsize=16)
      fig1.savefig(args.folder + '/train_results.pdf', bbox_inches='tight')
      plt.show(block=False)

    _,_ = valid_env.reset(roll_back=True)
    test_rewards_dict, test_acc_dict = agent.test(valid_env, agent.policy, args.num_test_episodes, render=args.render, step=args.step, quad_ftrs=args.quad_ftrs)

  else:
    if not args.model_file:
      raise ValueError, 'Model file not specified'
    policy.model.load_state_dict(torch.load(os.path.join(args.folder, args.model_file)))
    if args.model == 'linear': print(policy.model.fc.weight, policy.model.fc.bias)
    test_env = gym.make(args.valid_env)
    test_env.seed(args.seed_val)
    _, _ = test_env.reset()
    # raw_input('Weights loaded. Press enter to start testing...')
    test_rewards_dict, test_acc_dict = agent.test(test_env, policy, args.num_test_episodes, render=args.render, step=args.step, quad_ftrs=args.quad_ftrs)
  


  test_rewards = [it[1] for it in sorted(test_rewards_dict.items(), key=operator.itemgetter(0))]
  test_accs    = [it[1] for it in sorted(test_acc_dict.items(), key=operator.itemgetter(0))]
  test_results = dict()
  test_results['test_rewards'] = test_rewards_dict
  test_results['test_acc'] = test_acc_dict
  print ('Average test rewards = {}'.format(np.mean(test_rewards)))
  print('Dumping results')
  with open(os.path.abspath(args.folder) + '/test_results.json', 'w') as fp:
    json.dump(test_results, fp)

  print('(Testing) Number of episodes = {}, Total Reward = {}, Average reward = {},\
          Std. Dev = {}, Median reward = {}'.format(args.num_test_episodes, sum(test_rewards), \
          np.mean(test_rewards), np.std(test_rewards), np.median(test_rewards)))

  if args.plot:
    print('Plot test results and save')
    fig2, ax2 = plt.subplots(1,2)
    ax2[0].plot(range(len(test_rewards)), test_rewards)
    ax2[0].set_xlabel('Episodes')
    ax2[0].set_ylabel('Episode Reward')

    ax2[1].plot(range(len(test_accs)), test_accs)
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
  parser.add_argument("--expert", type=str, help='Expert to be used')
  parser.add_argument('--folder', required=True, type=str, help='Folder to load params.json and save results.json')
  parser.add_argument('--beta0', type=float, default=0.7, help='Value of mixing parameter after first episode of behavior cloning (decayed exponentially)')
  parser.add_argument('--alpha', type=float, default=0.01, help='Initial learning rate')
  parser.add_argument('--momentum', type=float, default=0.0, help='Momentum value for SGD')
  parser.add_argument('--gamma'    , type=float, default=1.0, help='Probability of keeping a data point')
  parser.add_argument('--batch_size', type=int, default=32, help='Batch size for SGD')
  parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for SGD')
  parser.add_argument('--weight_decay', type=float, default=0.2, help='L2 Regularization')
  parser.add_argument("--model_file", type=str, help="File to load model parameters during test phase")
  parser.add_argument('--render', action='store_true', help='Render during testing')
  parser.add_argument("--step", action='store_true', help='If true, then agent pauses before executing each action to show rendering')
  parser.add_argument("--use_cuda", action='store_true', help='Use cuda for training')
  parser.add_argument("--seed_val", type=int, required=True, help='Seed value for training')
  parser.add_argument("--re_init", action='store_true', help='Whether to re-initialize policy at every iteration of DAgger training')
  parser.add_argument("--test", action='store_true', help='Loads weights from the model file and runs testing')
  parser.add_argument("--plot", action='store_true', help='Whether to plot results or not')
  parser.add_argument("--heuristic", type=str, help="Heuristic policy to roll-in with. If not provided, expert will be used foll roll-in")
  parser.add_argument("--mixed_rollin", action='store_true', help="Roll-in with heuristic and oracle")
  parser.add_argument("--quad_ftrs", action='store_true', help="Use quadratic features or not")
  args = parser.parse_args()
  main(args)


