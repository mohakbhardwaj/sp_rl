#!/usr/bin/env python
import os, sys
sys.path.insert(0, os.path.abspath('../..'))
import gym
import sp_rl

env = gym.make('graphEnvToy-v0')
env.render()
#Run an episode with a random policy
done = False
while not done:
  action = env.action_space.sample()
  (obs, path), reward, done, info = env.step(action)
  print ('chosen edge = {}, obs = {}, reward = {}, done = {}'.format(env.edge_from_action(action), obs, reward, done))
  print [env.edge_from_action(a) for a in path]
  env.render()
  raw_input('Press enter for next step')