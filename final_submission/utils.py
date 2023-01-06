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
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 20):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 20 scores')
    plt.savefig(figure_file)


def compare_agents(x, ddpg_scores, td3_scores, figure_file):
    ddpg_running_avg = np.zeros(len(ddpg_scores))
    td3_running_avg = np.zeros(len(td3_scores))
    for i in range(len(ddpg_running_avg)):
        ddpg_running_avg[i] = np.mean(ddpg_scores[max(0, i - 20):(i + 1)])
        td3_running_avg[i] = np.mean(td3_scores[max(0, i - 20):(i + 1)])

    plt.plot(x, ddpg_running_avg, label="ddpg_avg_rewards")
    plt.plot(x, td3_running_avg, label="td3_avg_rewards")
    plt.title("Compare Agents")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(loc="upper left")
    plt.savefig(figure_file)


def train_agent(env, n_episodes, agent, figure_file):
    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_episodes):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-20:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.2f' % score,
              'trailing 20 games avg %.3f' % avg_score)

    x = [i + 1 for i in range(n_episodes)]
    plot_learning_curve(x, score_history, figure_file)

    return score_history
