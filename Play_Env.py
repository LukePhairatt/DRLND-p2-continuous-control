#!/usr/bin/env python
# coding: utf-8

#
# DDPG Continuous Control Project
#
from unityagents import UnityEnvironment
import numpy as np
import os
import torch
from ddpg_agent import Agent

# select this option to load with 20 agents) of the environment
unity_exe = os.path.join(os.getcwd(), "data/NoVis/Reacher20_Linux/Reacher.x86_64")
unity_env = UnityEnvironment(file_name=unity_exe)
# get the default brain
brain_name = unity_env.brain_names[0]
brain = unity_env.brains[brain_name]


#
# Inference
#

# Environment
env_info = unity_env.reset(train_mode=False)[brain_name]    # reset the environment
states = env_info.vector_observations                       # get the current state (for each agent)
state_size = state.shape[1]                                 # get a number of agents
action_size = brain.vector_action_space_size
num_agents = len(env_info.agents)
scores = np.zeros(num_agents)                               # initialize the score (for each agent)
# Agent
ddpg_agent = Agent(state_size=33, action_size=4, random_seed=42, device='gpu')
state_dict = torch.load('checkpoint_actor_final.pth')
ddpg_agent.actor_local.load_state_dict(state_dict)
# Play
while True:
    actions = ddpg_agent.act(states, add_noise=False)
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
unity_env.close()
