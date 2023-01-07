import numpy as np
import torch as T
import torch.nn.functional as F

from buffer import ReplayBuffer
from final_submission.utils import OUActionNoise
from networks import ActorNetwork, CriticNetwork


class BaseAgent:
    def __init__(self):
        self.memory = None

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)


class TD3Agent(BaseAgent):
    def __init__(self, actor_learning_rate, critic_learning_rate, input_shape, tau, env, gamma=0.99,
                 update_actor_interval=2, warmup=1000, n_actions=2,
                 max_size=1000000, layer1_size=400, layer2_size=300, batch_size=100, noise=0.1, agent_dir='tmp/td3'):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_shape, n_actions)
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.time_step_counter = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_interval = update_actor_interval
        self.noise = noise

        self.actor = ActorNetwork(actor_learning_rate, input_shape, layer1_size,
                                  layer2_size, n_actions=n_actions,
                                  name='actor', agent_dir=agent_dir, agent_type='_td3')
        self.critic_1 = CriticNetwork(critic_learning_rate, input_shape, layer1_size,
                                      layer2_size, n_actions=n_actions,
                                      name='critic_1', agent_dir=agent_dir, agent_type='_td3')
        self.critic_2 = CriticNetwork(critic_learning_rate, input_shape, layer1_size,
                                      layer2_size, n_actions=n_actions,
                                      name='critic_2', agent_dir=agent_dir, agent_type='_td3')

        self.target_actor = ActorNetwork(actor_learning_rate, input_shape, layer1_size,
                                         layer2_size, n_actions=n_actions,
                                         name='target_actor', agent_dir=agent_dir, agent_type='_td3')
        self.target_critic_1 = CriticNetwork(critic_learning_rate, input_shape, layer1_size,
                                             layer2_size, n_actions=n_actions,
                                             name='target_critic_1', agent_dir=agent_dir, agent_type='_td3')
        self.target_critic_2 = CriticNetwork(critic_learning_rate, input_shape, layer1_size,
                                             layer2_size, n_actions=n_actions,
                                             name='target_critic_2', agent_dir=agent_dir, agent_type='_td3')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        if self.time_step_counter < self.warmup:
            action = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)), device=self.actor.device)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            action = self.actor.forward(state).to(self.actor.device)

        action_prime = action + T.tensor(np.random.normal(scale=self.noise), dtype=T.float, device=self.actor.device)
        action_prime = T.clamp(action_prime, self.min_action[0], self.max_action[0])
        self.time_step_counter += 1
        return action_prime.cpu().detach().numpy()

    def learn(self):
        # Don't learn until there is at least the batch size in memory
        if self.memory.memory_counter < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        # Target Policy Smoothing
        target_actions = self.target_actor.forward(new_state)
        # Smoothing between -0.5 and 0.5
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        # In case that is beyond action space clamp again to min and max actions
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])

        q1_ = self.target_critic_1.forward(new_state, target_actions)
        q2_ = self.target_critic_2.forward(new_state, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        # Double Q Learning Update Rule
        critic_value = T.min(q1_, q2_)

        target = reward + self.gamma * critic_value
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_counter += 1

        # Only update actor every update actor interval
        if self.learn_step_counter % self.update_actor_interval != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau * critic_1_state_dict[name].clone() + (1 - tau) * \
                                        target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau * critic_2_state_dict[name].clone() + (1 - tau) * \
                                        target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_state_dict[
                name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_actor()
        self.target_actor.save_actor()
        self.critic_1.save_critic()
        self.critic_2.save_critic()
        self.target_critic_1.save_critic()
        self.target_critic_2.save_critic()

    def load_models(self):
        self.actor.load_actor()
        self.target_actor.load_actor()
        self.critic_1.load_critic()
        self.critic_2.load_critic()
        self.target_critic_1.load_critic()
        self.target_critic_2.load_critic()


class DDPGAgent(BaseAgent):
    def __init__(self, actor_learning_rate, critic_learning_rate, input_shape, tau, n_actions, gamma=0.99,
                 max_size=1000000, layer1_size=400,
                 layer2_size=300, batch_size=64, agent_dir='tmp/ddpg'):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.memory = ReplayBuffer(max_size, input_shape, n_actions)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(actor_learning_rate, input_shape, layer1_size, layer2_size,
                                  n_actions=n_actions, name='actor', agent_dir=agent_dir, agent_type='_ddpg')
        self.critic = CriticNetwork(critic_learning_rate, input_shape, layer1_size, layer2_size,
                                    n_actions=n_actions, name='critic', agent_dir=agent_dir, agent_type='_ddpg')

        self.target_actor = ActorNetwork(actor_learning_rate, input_shape, layer1_size, layer2_size,
                                         n_actions=n_actions, name='target_actor', agent_dir=agent_dir,
                                         agent_type='_ddpg')

        self.target_critic = CriticNetwork(critic_learning_rate, input_shape, layer1_size, layer2_size,
                                           n_actions=n_actions, name='target_critic', agent_dir=agent_dir,
                                           agent_type='_ddpg')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        # Must put actor in eval mode
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        action = self.actor.forward(state).to(self.actor.device)
        action_prime = action + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        # And then put back in train mode
        self.actor.train()

        return action_prime.cpu().detach().numpy()[0]

    def learn(self):
        # Don't learn until there is at least the batch size in memory
        if self.memory.memory_counter < self.batch_size:
            return

        states, actions, rewards, new_states, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        new_states = T.tensor(new_states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        target_actions = self.target_actor.forward(new_states)
        critic_value_ = self.target_critic.forward(new_states, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma * critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1 - tau) * target_critic_state_dict[
                name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_state_dict[
                name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_actor()
        self.target_actor.save_actor()
        self.critic.save_critic()
        self.target_critic.save_critic()

    def load_models(self):
        self.actor.load_actor()
        self.target_actor.load_actor()
        self.critic.load_critic()
        self.target_critic.load_critic()
