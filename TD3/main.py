import gym

from agents import TD3Agent, DDPGAgent
from utils import *

# TODO: Using cuda isn't currently working
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

# Hyper Parameters
hidden_layer_1 = 350  # Size of first Hidden Layer
hidden_layer_2 = 350  # Size of second Hidden Layer
num_of_hidden_layers = 2  # 1 if we want 1 Hidden Layer, 2 if we want 2 Hidden Layers
actor_lr = 1e-4  # Actor learning rate
critic_lr = 1e-3  # Critic learning rate
gamma = 0.99  # Discounting factor
tau = 0.005  # Target update factor
memory_size = 50000  # Memory size
number_of_elements = 5000  # Number of elements in memory
num_episodes = 2000  # Training episodes
batch_size = 64  # Batch size
eval_episodes = 5  # Number of evaluation episodes
popicy_delay = 2  # Frequency we want to update the policy
ddpg_file = 'DDPG_Agent'  # Name we want to save
td3_file = 'TD3_Agent'  # Name we want to save
directory = 'saves'  # Directory where we want to save

ddpg_env = gym.make('LunarLander-v2', continuous=True)
ddpg_env = gym.wrappers.RecordVideo(ddpg_env, f"videos/{ddpg_file}")

td3_env = gym.make('LunarLander-v2', continuous=True)
td3_env = gym.wrappers.RecordVideo(td3_env, f"videos/{td3_file}")

# Create Agents
ddpg_agent = DDPGAgent(ddpg_env, hidden_layer_1, hidden_layer_2, actor_lr, critic_lr, gamma, tau, memory_size, num_of_hidden_layers,
                       device)
td3_agent = TD3Agent(td3_env, hidden_layer_1, hidden_layer_2, actor_lr, critic_lr, gamma, tau, memory_size, num_of_hidden_layers,
                     popicy_delay, device)

# Train Agents
fill_memory(ddpg_env, number_of_elements, ddpg_agent)
ddpg_avg_rewards = train(ddpg_agent, ddpg_env, num_episodes, batch_size, ddpg_file, directory)
print()

fill_memory(td3_env, number_of_elements, td3_agent)
td3_avg_rewards = train(td3_agent, td3_env, num_episodes, batch_size, td3_file, directory)
print()

# Evaluate Agents
ddpg_agent.load(ddpg_file, directory)
evaluate_agent(ddpg_agent, ddpg_env, eval_episodes)
print()

td3_agent.load(td3_file, directory)
evaluate_agent(td3_agent, td3_env, eval_episodes)
print()

# Compare Agents
plt.plot(ddpg_avg_rewards, label="ddpg_avg_rewards")
plt.plot(td3_avg_rewards, label="td3_avg_rewards")
plt.title("Compare Agents")
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc="upper left")
plt.show()
