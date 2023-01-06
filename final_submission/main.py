import gym

from agents import DDPGAgent, TD3Agent
from utils import train_agent, compare_agents

if __name__ == '__main__':
    ddpg_env = gym.make('LunarLander-v2', continuous=True)
    ddpg_env = gym.wrappers.RecordVideo(ddpg_env, f"videos/DDPG_agent")

    td3_env = gym.make('LunarLander-v2', continuous=True)
    td3_env = gym.wrappers.RecordVideo(td3_env, f"videos/TD3_agent")

    alpha = 0.0001
    beta = 0.001
    observation_space = ddpg_env.observation_space.shape
    tau = 0.001
    batch_size = 64
    layer1_size = 400
    layer2_size = 300
    action_space = ddpg_env.action_space.shape[0]
    n_episodes = 2

    # DDPG
    ddpg_agent = DDPGAgent(alpha=alpha, beta=beta,
                           input_dims=observation_space, tau=tau,
                           batch_size=batch_size, layer1_size=layer1_size, layer2_size=layer2_size,
                           n_actions=action_space)
    ddpg_filename = 'LunarLander_alpha_' + str(alpha) + '_beta_' + \
                    str(beta) + '_' + str(n_episodes) + '_games' + '_DDPG'
    ddpg_figure_file = 'plots/' + ddpg_filename + '.png'

    ddpg_scores = train_agent(ddpg_env, n_episodes, ddpg_agent, ddpg_figure_file)

    print()
    print()

    # TD3
    td3_agent = TD3Agent(alpha=alpha, beta=beta,
                         input_dims=observation_space, tau=tau,
                         env=td3_env, batch_size=batch_size, layer1_size=layer1_size, layer2_size=layer2_size,
                         n_actions=action_space)

    td3_filename = 'LunarLander_alpha_' + str(alpha) + '_beta_' + \
                   str(beta) + '_' + str(n_episodes) + '_games' + '_TD3'
    td3_figure_file = 'plots/' + td3_filename + '.png'

    td3_scores = train_agent(td3_env, n_episodes, td3_agent, td3_figure_file)

    # Comparison
    x = [i + 1 for i in range(n_episodes)]
    comparison_filename = 'LunarLander_alpha_' + str(alpha) + '_beta_' + \
                   str(beta) + '_' + str(n_episodes) + '_games' + '_comparison'
    comparison_figure_file = 'plots/' + comparison_filename + '.png'
    compare_agents(x, ddpg_scores, td3_scores, comparison_figure_file)
