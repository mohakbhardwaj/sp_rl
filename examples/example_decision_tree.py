#!/usr/bin/env python
import os,sys
sys.path.insert(0, os.path.abspath('..'))
import matplotlib.pyplot as plt
import numpy as np
import gym
import sp_rl
from sp_rl.agents import DecisionTreeAgent


env = gym.make('graphEnvToy-v0')
env.seed(0)
np.random.seed(0)

agent = DecisionTreeAgent(env)
