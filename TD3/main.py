import gym
import torch

from agents import TD3Agent, DDPGAgent
from utils import *

env = gym.make('Pendulum-v1')

# Seed
seed = 10
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Hyper Parameters
hidden_l1 = 350  # Size of first Hidden Layer
hidden_l2 = 350  # Size of second Hidden Layer
num_of_hd_layers = 2  # 1 if we want 1 Hidden Layer, 2 if we want 2 Hidden Layers
actor_lr = 1e-4  # Actor learning rate
critic_lr = 1e-3  # Critic learning rate
gamma = 0.99  # Discounting factor
tau = 1e-2  # Target update factor
memory_size = 50000  # Memory size
number_of_elements = 5000  # Number of elements in memory
num_episodes = 500  # Training episodes
batch_size = 64  # Batch size
eval_episodes = 5  # Number of evaluation episodes
popicy_delay = 2  # Frequency we want to update the policy
ddpg_file = 'DDPG_Agent1'  # Name we want to save
td3_file = 'TD3_Agent1'  # Name we want to save
directory = 'saves'  # Directory where we want to save

# Create Agents
ddpg_agent = DDPGAgent(env, hidden_l1, hidden_l2, actor_lr, critic_lr, gamma, tau, memory_size, num_of_hd_layers)
td3_agent = TD3Agent(env, hidden_l1, hidden_l2, actor_lr, critic_lr, gamma, tau, memory_size, num_of_hd_layers,
                     popicy_delay)

# Train Agents
fill_memory(env, number_of_elements, ddpg_agent)
ddpg_avg_rewards = train(ddpg_agent, env, num_episodes, batch_size, ddpg_file, directory)

fill_memory(env, number_of_elements, td3_agent)
td3_avg_rewards = train(td3_agent, env, num_episodes, batch_size, td3_file, directory)

# Evaluate Agents
ddpg_agent.load(ddpg_file, directory)
evaluate_agent(ddpg_agent, env, eval_episodes, render=True)

td3_agent.load(td3_file, directory)
evaluate_agent(td3_agent, env, eval_episodes, render=True)

# Compare Agents
plt.plot(ddpg_avg_rewards, label="ddpg_avg_rewards")
plt.plot(td3_avg_rewards, label="td3_avg_rewards")
plt.title("Compare Agents")
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc="upper left")
