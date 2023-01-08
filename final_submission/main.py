import gym

from agents import DDPGAgent, TD3Agent
from utils import train_agent, compare_agents


def train_agents():
    ddpg_env = gym.make('LunarLander-v2', continuous=True)
    ddpg_env = gym.wrappers.RecordVideo(ddpg_env, f"videos/DDPG_agent")

    td3_env = gym.make('LunarLander-v2', continuous=True)
    td3_env = gym.wrappers.RecordVideo(td3_env, f"videos/TD3_agent")

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    tau = 0.005
    batch_size = 64
    layer1_size = 400
    layer2_size = 300
    input_shape = ddpg_env.observation_space.shape
    n_actions = ddpg_env.action_space.shape[0]
    n_episodes = 2000

    # DDPG
    ddpg_agent = DDPGAgent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate,
                           input_shape=input_shape, tau=tau, env=ddpg_env,
                           batch_size=batch_size, layer1_size=layer1_size, layer2_size=layer2_size,
                           n_actions=n_actions)
    ddpg_filename = 'LunarLander_actor_learning_rate_' + str(actor_learning_rate) + '_critic_learning_rate_' + str(
        critic_learning_rate) + '_' + str(n_episodes) + '_games' + '_DDPG'
    ddpg_figure_file = 'plots/' + ddpg_filename + '.png'

    ddpg_scores = train_agent(ddpg_env, n_episodes, ddpg_agent, ddpg_figure_file, 'ddpg')

    print()
    print()

    # TD3
    td3_agent = TD3Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate,
                         input_shape=input_shape, tau=tau,
                         env=td3_env, batch_size=batch_size, layer1_size=layer1_size, layer2_size=layer2_size,
                         n_actions=n_actions)

    td3_filename = 'LunarLander_actor_learning_rate_' + str(actor_learning_rate) + '_critic_learning_rate_' + str(
        critic_learning_rate) + '_' + str(n_episodes) + '_games' + '_TD3'
    td3_figure_file = 'plots/' + td3_filename + '.png'

    td3_scores = train_agent(td3_env, n_episodes, td3_agent, td3_figure_file, 'td3')

    # Learning Comparison
    x = [i + 1 for i in range(n_episodes)]

    comparison_filename = 'LunarLander_actor_learning_rate_' + str(
        actor_learning_rate) + '_critic_learning_rate_' + str(critic_learning_rate) + '_' + str(
        n_episodes) + '_games' + '_comparison'
    comparison_figure_file = 'plots/' + comparison_filename + '.png'
    compare_agents(x, ddpg_scores, td3_scores, comparison_figure_file)

    # Evaluation
    ddpg_env = gym.make('LunarLander-v2', continuous=True)
    td3_env = gym.make('LunarLander-v2', continuous=True)

    n_evaluation_episodes = 200
    x = [i + 1 for i in range(n_evaluation_episodes)]

    ddpg_figure_file = 'plots/' + ddpg_filename + '_eval' + '.png'
    ddpg_scores = train_agent(ddpg_env, n_evaluation_episodes, ddpg_agent, ddpg_figure_file, 'ddpg', evaluate=True)

    print()
    print()

    td3_figure_file = 'plots/' + td3_filename + '_eval' + '.png'
    td3_scores = train_agent(td3_env, n_evaluation_episodes, td3_agent, td3_figure_file, 'td3', evaluate=True)

    # Evaluation Comparison
    comparison_figure_file = 'plots/' + comparison_filename + '_eval' + '.png'
    compare_agents(x, ddpg_scores, td3_scores, comparison_figure_file)


def train_random_agent():
    env = gym.make('LunarLander-v2', continuous=True)
    n_episodes = 500
    train_agent(env, n_episodes, None, 'plots/random_agent.png', random_action=True)


if __name__ == '__main__':
    do_train_agents = True
    do_train_random_agent = False

    if do_train_agents:
        train_agents()

    if do_train_random_agent:
        train_random_agent()

    print()
    print('Done')
