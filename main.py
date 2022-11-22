import numpy as np
import torch
import gym
import pybullet_envs
import argparse
import os
import time

import ReplayBuffer
import TD3


def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print(eval_episodes, avg_reward)
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3")  # Policy name
    parser.add_argument("--env_name", default="HalfCheetahBulletEnv-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--filename",
                        default="TD3_HalfCheetahBulletEnv-v0_0")  # Model filename prefix e.g. "TD3_HalfCheetahBulletEnv-v0_0"
    parser.add_argument("--eval_episodes", default=1, type=int)  # Evaluation episodes
    parser.add_argument("--visualize", action="store_true")  # Visualize or not

    args = parser.parse_args()

    filename = args.filename

    print
    "---------------------------------------"
    print
    "Loading model from: %s" % (filename)
    print
    "---------------------------------------"

    env = gym.make(args.env_name)
    if args.visualize:
        env.render(mode="human")

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if args.policy_name == "TD3":
        policy = TD3.TD3(state_dim, action_dim, max_action)

    # Load model
    policy.load(filename, './pytorch_models/')

    # Start evaluation
    _ = evaluate_policy(policy, eval_episodes=args.eval_episodes, visualize=args.visualize)