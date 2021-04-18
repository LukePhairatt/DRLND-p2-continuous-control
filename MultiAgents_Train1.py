#!/usr/bin/env python
# coding: utf-8

#
# DDPG Continuous Control Project
#
from unityagents import UnityEnvironment
import numpy as np
import os
import torch
from collections import deque
from ddpg_agent import Agent, training_ddpg
import matplotlib.pyplot as plt

# select this option to load with 20 agents) of the environment
unity_exe = os.path.join(os.getcwd(), "data/NoVis/Reacher20_Linux/Reacher.x86_64")
unity_env = UnityEnvironment(file_name=unity_exe)


#
# Train DDPG
#

# Train 1- cpu: number of BATCH update (num_sample_update) = 1
# Create agent
ddpg_agent = Agent(state_size=33, action_size=4, random_seed=42, device='cpu')
# Train
all_scores = training_ddpg(unity_env, ddpg_agent, chk_prefix="1", episodes=200, print_every=10, num_sample_update=1)
# Plot
plt.figure()
plt.xlabel('Episode')
plt.ylabel('Scores')
plt.plot(all_scores)
plt.savefig('agent20_train1.png')
plt.show()
unity_env.close()
