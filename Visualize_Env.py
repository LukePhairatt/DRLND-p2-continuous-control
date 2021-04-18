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
from ddpg_agent import Agent
import matplotlib.pyplot as plt

# select this option to load with 20 agents) of the environment
unity_exe = os.path.join(os.getcwd(), "data/NoVis/Reacher20_Linux/Reacher.x86_64")
env = UnityEnvironment(file_name=unity_exe)

"""
# get the default brain
# brain_name = env.brain_names[0]
# brain = env.brains[brain_name]


#
# Examine the State and Action Spaces
#
# In this environment, a double-jointed arm can move to target locations.
# A reward of `+0.1` is provided for each step that the agent's hand is in the goal location.
# Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

# The observation space consists of `33` variables corresponding to position, rotation, velocity,
# and angular velocities of the arm.
# Each action is a vector with four numbers, corresponding to torque applicable to two joints.
# Every entry in the action vector must be a number between `-1` and `1`.
env_info = env.reset(train_mode=True)[brain_name]
# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)
# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)
# examine the state space
state = env_info.vector_observations
state_size = state.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(state.shape[0], state_size))
print('The state for the first agent looks like:', state[0])
"""

#
# Train DDPG
#

def ddpg(env, agent, chk_prefix='1', episodes=500, print_every=10):
    all_scores = []
    avg_scores_window = []
    noise_damp = 0
    scores_window = deque(maxlen=100)
    actor_checkpoint = "checkpoint_actor_" + chk_prefix + ".pth"
    critic_checkpoint = "checkpoint_critic_" + chk_prefix + ".pth"
    brain_name = env.brain_names[0]
    num_agents = 20
    env_info = env.reset(train_mode=True)[brain_name]
    for i_episode in range(1, episodes+1):
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        while True:
            actions = agent.act(states, noise_damp)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            agent.step(states, actions, rewards, next_states, dones, num_updates=1)
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        avg_score = np.mean(scores)
        scores_window.append(avg_score)
        all_scores.append(avg_score)
        avg_scores_window.append(np.mean(scores_window))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), actor_checkpoint)
            torch.save(agent.critic_local.state_dict(), critic_checkpoint)
        if np.mean(scores_window) >= 30:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), actor_checkpoint)
            torch.save(agent.critic_local.state_dict(), critic_checkpoint)
            break
    return all_scores

# Train 1- cpu: Create the agent
agent = Agent(state_size=33, action_size=4, random_seed=42, device='cpu')
all_scores = ddpg(env, agent, chk_prefix="1", episodes=100, print_every=10)
plt.figure()
plt.xlabel('Episode')
plt.ylabel('Scores')
plt.plot(all_scores)
plt.show()
plt.savefig('agent20_train1.png')
env.close()
