import gym
from gym import error, spaces, utils

from agents import DDPG_Agent, TD3_Agent
from utils import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from gym.utils import seeding
i
env = gym.make('Pendulum-v1')

###### SEED  ######
seed = 10
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
##################

########## *** PARAMETERS *** ##########
hidden_l1 = 350 # Size of first Hidden Layer
hidden_l2 = 350 # Size of second Hidden Layer
num_of_hd_layers = 2 # 1 if we want 1 Hidden Layer, 2 if we want 2 Hidden Layers
actor_lr = 1e-4 # Actor learning rate
critic_lr = 1e-3 # Critic learning rate
gamma = 0.99 # Discounting factor
tau = 1e-2 # Target update factor
memory_size = 50000 # Memory size
number_of_elements = 5000 # Number of elements in memory
num_episodes = 50 # Training episodes
batch_size = 64 # Batch size
eval_episodes = 5 # Number of evaluation episodes
popicy_delay = 2 # Frequency we want to update the policy
file2save = 'DDPG_Agent1' # Name we want to save
# file2save = 'TD3_Agent1' # Name we want to save
directory = 'saves' # Directory where we want to save

# agent = DDPG_Agent(env, hidden_l1, hidden_l2, actor_lr, critic_lr, gamma, tau, memory_size, num_of_hd_layers)
agent = TD3_Agent(env, hidden_l1, hidden_l2, actor_lr, critic_lr, gamma, tau, memory_size, num_of_hd_layers, popicy_delay)

####### TRAIN AGENT #######
fill_memory(env, number_of_elements, agent)
train(agent, env, num_episodes, batch_size, file2save, directory)

######### EVALUATE AGENT #########
agent.load(file2save, directory)
evaluate_agent(agent, env, eval_episodes, render=True)

"""If we want to train multiple agents with different parameters and compare their results,
we can store 'avg_rewards' in some variable and plot them"""

# agent1 = DDPG_Agent(env, 250, 250, 1e-4, 1e-3, 0.99, 0.01, 50000, 2)
# agent2 = DDPG_Agent(env, 350, 350, 1e-4, 1e-3, 0.99, 0.01, 50000, 2)
# agent3 = TD3_Agent(env, 350, 350, 1e-4, 1e-3, 0.99, 0.01, 50000, 2, 2)

# agent1_average = train(agent1, env, 50, 64, 'agent1_ddpg', 'saves')
# agent2_average = train(agent2, env, 50, 64, 'agent2_ddpg', 'saves')
# agent3_average = train(agent3, env, 50, 64, 'agent3_td3', 'saves')

# plt.plot(agent1_average, label = "agent1")
# plt.plot(agent2_average, label = "agent2")
# plt.plot(agent3_average, label = "agent3")
# plt.title("Compare Agents")
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.legend(loc="upper left")