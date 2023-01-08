import matplotlib.pyplot as plt
import numpy as np


class OUActionNoise:
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


def plot_rewards(x, scores, figure_file, label):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 20):(i + 1)])

    fig, ax = plt.subplots()
    ax.plot(x, running_avg, label=label)
    fig.suptitle('Running average of previous 20 scores')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    fig.legend(loc="upper left")
    fig.savefig(figure_file)


def compare_agents(x, ddpg_scores, td3_scores, figure_file):
    ddpg_running_avg = np.zeros(len(ddpg_scores))
    td3_running_avg = np.zeros(len(td3_scores))
    for i in range(len(ddpg_running_avg)):
        ddpg_running_avg[i] = np.mean(ddpg_scores[max(0, i - 20):(i + 1)])
        td3_running_avg[i] = np.mean(td3_scores[max(0, i - 20):(i + 1)])

    fig, ax = plt.subplots()
    ax.plot(x, ddpg_running_avg, label="ddpg_avg_rewards")
    ax.plot(x, td3_running_avg, label="td3_avg_rewards")
    fig.suptitle("Compare Agents")
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    fig.legend(loc="upper left")
    fig.savefig(figure_file)


def train_agent(env, n_episodes, agent, figure_file, label, random_action=False, evaluate=False):
    if evaluate:
        agent.load_models()

    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_episodes):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            if not random_action and not evaluate:
                action = agent.choose_action(observation)
                next_observation, reward, done, info = env.step(action)
                agent.remember(observation, action, reward, next_observation, done)
                agent.learn()
                score += reward
                observation = next_observation
            elif random_action:
                action = env.action_space.sample()
                next_observation, reward, done, info = env.step(action)
                score += reward
                observation = next_observation
            elif evaluate:
                action = agent.choose_action(observation)
                next_observation, reward, done, info = env.step(action)
                score += reward
                observation = next_observation

        score_history.append(score)
        avg_score = np.mean(score_history[-20:])

        if avg_score > best_score and not random_action and not evaluate:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.2f' % score,
              'trailing 20 games avg %.3f' % avg_score)

    x = [i + 1 for i in range(n_episodes)]
    plot_rewards(x, score_history, figure_file, label)

    return score_history
