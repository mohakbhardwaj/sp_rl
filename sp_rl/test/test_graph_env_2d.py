#!/usr/bin/env python
import os, sys
sys.path.insert(0, os.path.abspath('../..'))
import gym
import sp_rl
from sp_rl.envs import GraphEnv2D

env = gym.make('graphEnv2D-v0')
obs, graph_info = env.reset()
env.seed(0)
env.render()
#Run an episode with a random policy
done = False
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
  print ('chosen edge = {}, obs = {}, reward = {}, done = {}'.format(env.edge_from_action(action), obs, reward, done))
  # print [env.edge_from_action(a) for a in path]
  # print obs[action], action
  env.render()
  raw_input('Press enter for next step')