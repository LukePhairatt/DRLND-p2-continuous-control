# Project 1: Navigation

### Introduction
DQN and Double DQN have been used for learning by the agent (see dqn_agents.py). In this repository, there are 3 files used for training the agents as follows.  

**Navigation.ipynb** is the main routine to create the environment and the agen. It is where the agent interacts with the environment during learning and keeping track of scores for each episode.  

**dqn_agent** is the DQN algorithm to update the Q network in both a local and a target network.  

**model.py** is the Q network definition.   


### Learning  
**Hyperparametrs**  
```
BUFFER_SIZE = int(1e5)  # replay buffer size   
BATCH_SIZE = 64         # minibatch size  
GAMMA = 0.99            # discount factor  
TAU = 1e-3              # for soft update of target parameters  
LR = 5e-4               # learning rate   
UPDATE_EVERY = 4        # how often to update the network  

max_t = 1000		# maximum number of timesteps per episode  
eps_start = 1.0		# starting value of epsilon, for epsilon-greedy action selection  
eps_end = 0.01          # minimum value of epsilon  
eps_decay = 0.995       # multiplicative factor (per episode) for decreasing epsilon  
```

**Algorithm**  
DQN learning is acheived by computing loss with Q target and Q expected by this formula  

`line 100-113 in dqn_agent.py`
```
Q_target(st, at) = reward + gamma * max(Q_target(st+1))  
Q_expected(st, at) = Q_local(st, at)

where 
	st = current state,  st+1 = next state
	at = current action, at+1 = next action

```
Fixed Q_target update is done by a soft-update with a TAU factor `line 118-129 in dqn_agent.py`.  

**Q Network**  
The network is defined in `model.py`. It has 3 fully connected layers with RELU activation (input and hidden layer).  
By default, it has input and output of each layer as follows.  

```
input layer:  [state_size, 128]  
hidden layer: [128, 64]  
output layer: [64, action_size]   
```

### Plot of Rewards    

**DQN Scores**  
![DQN_scores](/images/DQN_scores.png)  

```
Episode 100	Average Score: 0.08
Episode 200	Average Score: 2.04
Episode 300	Average Score: 5.38
Episode 400	Average Score: 8.57
Episode 500	Average Score: 12.80
Episode 504	Average Score: 13.00
Environment solved in 504 episodes!	Average Score: 13.00
```

**Double DQN Scores**  
![double_DQN_scores](/images/double_DQN_scores.png)  
```
Episode 100	Average Score: 0.96
Episode 200	Average Score: 4.24
Episode 300	Average Score: 7.54
Episode 400	Average Score: 10.69
Episode 482	Average Score: 13.05
Environment solved in 482 episodes!	Average Score: 13.05
```
### Future Improvements
Double DQN might not out perform normal DQN is some environment. There are a few approaches such as **Dueling DQN**, and   
**Prioritized Experience Replay** or **a combination of these approaches** have been shown with an improvement toward  
a stability of learning. The brief ideas for the future improvement of these approaches are as follows.

With the problem of uniformly sampling experiences from a reply memory regardless of their significance, some rare important experiences may not be picked as frequently as we hope. `Prioritized Experience Replay` addresses this issue by prioritizing the experience transition in the replay memory. The more important experience the agent can learn from more effectively (e.g. bigger TD error) the higher priority for sampling will be assigned.   
 
`Dueling DQN` focuses on the improvement of Q-values estimation. The core idea is to use 2 streams of branched-off DQN to estimate 1. state values- V(s) and 2. Advantage values (A(s, a)). The value function V(s) indicates a reward on a given state. And the advantage function A(s, a) indicates an advantage one action over the other actions on a given state. These 2 values are then combined to estimate the better Q-values.  


