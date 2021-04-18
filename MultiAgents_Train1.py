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

#
# Train DDPG
#

def ddpg(env, agent, chk_prefix='1', episodes=500, print_every=10, num_samples=1):
    all_scores = []
    avg_scores_window = []
    scores_window = deque(maxlen=100)
    actor_checkpoint = "checkpoint_actor_" + chk_prefix + ".pth"
    critic_checkpoint = "checkpoint_critic_" + chk_prefix + ".pth"
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    for i_episode in range(1, episodes+1):
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            agent.step(states, actions, rewards, next_states, dones, num_samples=num_samples)
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
# Create agent
agent = Agent(num_agents=20, state_size=33, action_size=4, random_seed=42, device='cpu')
# Train
all_scores = ddpg(env, agent, chk_prefix="1", episodes=200, print_every=10, num_samples=1)
# Plot
plt.figure()
plt.xlabel('Episode')
plt.ylabel('Scores')
plt.plot(all_scores)
plt.savefig('agent20_train1.png')
plt.show()
env.close()
