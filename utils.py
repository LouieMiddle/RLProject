import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque


class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)  # Some kind of "lists" optimized for fast fixed-length operations

    def push(self, state, action, new_state, reward, done):
        """Fill the buffer with status from enviroment"""

        experience = (state, action, new_state, np.array([reward]), done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Random samples with 'batch_size' elements"""
        state_batch = []
        action_batch = []
        new_state_batch = []
        reward_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, new_state, reward, done = experience
            state_batch.append(state)
            action_batch.append(action)
            new_state_batch.append(new_state)
            reward_batch.append(reward)
            done_batch.append(done)

        return state_batch, action_batch, new_state_batch, reward_batch, done_batch

    def reset(self):
        # Clear the buffer
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class OUNoise(object):
    """The Ornstein-Uhlenbeck Process generates noise that is correlated with the previous noise,
    as to prevent the noise from canceling out or “freezing” the overall dynamics
    https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process

    OUNoise copied from here: https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
    """

    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)

        return np.clip(action + ou_state, self.low, self.high)


def fill_memory(env, number_of_elements, agent):
    """Fill the Memory with some random elements = 'number_of_elements' """
    time_steps = 0
    state = env.reset()
    done = False

    while time_steps < number_of_elements:
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)

        agent.memory.push(state, action, new_state, reward, done)
        state = new_state
        time_steps += 1
        if done:
            state = env.reset()
            done = False
    print("Added %i elements in Memory" % number_of_elements)


def evaluate_agent(agent, env, eval_episodes, render=False):
    """Evaluate performance of a policy after training"""
    avg_reward = 0
    for i in range(eval_episodes):
        obs = env.reset()
        done = False
        print("Current episode: ", i + 1)
        while not done:
            if render:
                env.render()
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes

    print("\n---------------------------------------")
    print("Evaluation over {:d} episodes: {:f}".format(eval_episodes, avg_reward))
    print("---------------------------------------")
    env.close()


def train(agent, env, num_episodes, batch_size, file2save, directory):
    rewards = []
    avg_rewards = []
    best_avg = -5000
    avg_reward = 0
    steps_done = 0
    best_avg_reward = -5000

    for i_episode in range(num_episodes):
        episode_reward = 0
        step = 0
        state = env.reset()
        agent.ns.reset()
        while True:
            action = agent.select_action(state, step, noise=True)
            new_state, reward, done, info = env.step(action)

            agent.memory.push(state, action, new_state, reward, done)
            agent.update(batch_size, step)

            state = new_state
            episode_reward += reward
            step += 1

            avg_reward = np.mean(rewards[-10:])
            if done:
                print("Episode: ", i_episode + 1, " Reward: ", np.round(episode_reward, decimals=2), \
                      " Average reward: ", np.round(avg_reward, decimals=2))

                if best_avg <= avg_reward:  # Save when we have best average reward
                    best_avg = avg_reward
                    agent.save(file2save, directory)
                break

        rewards.append(episode_reward)
        avg_rewards.append(avg_reward)

        if best_avg_reward < avg_reward:
            best_avg_reward = avg_reward
    print("\n\n\nBest average reward in {:d}, episodes is: {:f}".format((i_episode + 1),
                                                                        np.round(best_avg_reward, decimals=2)))
    plt.plot(rewards, label="Rewards")
    plt.plot(avg_rewards, label="Avg Rewards")
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Average Reward')
    plt.legend(loc="upper left")
    plt.show()

    return avg_rewards