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
'''
def training_ddpg(env, agent, chk_prefix='1', episodes=500, print_every=10):
    scores_all = []
    scores_window = deque(maxlen=100)
    actor_checkpoint = "checkpoint_actor_" + chk_prefix + ".pth"
    critic_checkpoint = "checkpoint_critic_" + chk_prefix + ".pth"
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    for i_episode in range(1, episodes + 1):
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        while True:
            # interacting
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            # learning
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        mean_score = np.mean(scores)
        scores_window.append(mean_score)
        scores_all.append(mean_score)
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
    return scores_all
'''

# Train 2- gpu: number of BATCH update (num_sample_update) = 2
# Create agent and load check point from the previous training
ddpg_agent = Agent(state_size=33, action_size=4, random_seed=42, device='gpu')
state_dict = torch.load('checkpoint_actor_1.pth')
ddpg_agent.actor_local.load_state_dict(state_dict)
# Train
all_scores = training_ddpg(unity_env, ddpg_agent, chk_prefix="final", episodes=500, print_every=10, num_sample_update=2)
# Plot
plt.figure()
plt.xlabel('Episode')
plt.ylabel('Scores')
plt.plot(all_scores)
plt.savefig('agent20_final.png')
plt.show()
unity_env.close()
