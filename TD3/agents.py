import torch.autograd
import torch.optim as optim
from torch.autograd import Variable

from networks import *
from utils import OUNoise, Memory


class DDPGAgent(object):
    def __init__(self, env, hidden_l1, hidden_l2, actor_lr, critic_lr, gamma, tau, memory_size, num_of_hd_layers):
        self.gamma = gamma
        self.tau = tau
        self.num_actions = env.action_space.shape[0]
        self.num_states = env.observation_space.shape[0]
        self.ns = OUNoise(env.action_space)

        if num_of_hd_layers == 1:
            self.actor = DDPGActor1(self.num_states, hidden_l1, self.num_actions)
            self.critic = DDPGCritic1(self.num_actions + self.num_states, hidden_l1)  # output setat 1 by defauld
            self.actor_target = DDPGActor1(self.num_states, hidden_l1, self.num_actions)
            self.critic_target = DDPGCritic1(self.num_actions + self.num_states, hidden_l1)
        elif num_of_hd_layers == 2:
            self.actor = DDPGActor2(self.num_states, hidden_l1, hidden_l2, self.num_actions)
            self.critic = DDPGCritic2(self.num_actions + self.num_states, hidden_l1, hidden_l2)
            self.actor_target = DDPGActor2(self.num_states, hidden_l1, hidden_l2, self.num_actions)
            self.critic_target = DDPGCritic2(self.num_actions + self.num_states, hidden_l1, hidden_l2)
        else:
            print("Wrong number of hidden layers. Use 1 or 2.")

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.memory = Memory(memory_size)
        self.loss_func = nn.MSELoss()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state, step=0, noise=False):
        """state must be a tensor
           output action must be a float number

           *** IN TRAINING MODE SET NOISE = TRUE
        """

        state = torch.Tensor(state)
        action = self.actor.forward(state)
        action = action.detach().numpy()

        if noise:
            action = self.ns.get_action(action, step)
        return action

    def update(self, batch_size, step):
        # !!! I added 'step' parameter to use same 'train()' method to train both agents (DDPG and TD3)
        # Here 'step' parameter is never used

        # We will wait until our memory buffer has at least "batch_size" position
        if self.memory.__len__() < batch_size:
            return

        # Import samples from memory
        state, action, new_state, reward, done = self.memory.sample(batch_size)
        # If the network learned only from consecutive samples of experience as they occurred
        # sequentially in the environment, the samples would be highly correlated and would
        # therefore lead to inefficient learning. Taking random samples from replay memory breaks this correlation.

        # Convert samples to Tensors
        action = torch.FloatTensor(action)
        state = torch.FloatTensor(state)
        new_state = torch.FloatTensor(new_state)
        reward = torch.FloatTensor(reward)
        done = torch.Tensor(done)

        new_action = self.actor_target(new_state).detach()  # mu_target
        Q_target = self.critic_target(new_state, new_action)  # Q_target
        target = reward + (1 - done) * self.gamma * Q_target  # y(r,s',d)

        Q_val = self.critic(state, action)  # Q_network

        # y(r,s',d) = r + gamma * (1-d) * Q_targ(s',mu_targ(s'))

        critic_loss = self.loss_func(Q_val, target)  # Compute Loss Function

        actor_loss = -self.critic(state, self.actor(state)).mean()  # Compute Loss Function

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update target network
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + (1 - self.tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


class TD3Agent(object):
    def __init__(self, env, hidden_l1, hidden_l2, actor_lr, critic_lr, gamma, tau, memory_size, num_of_hd_layers,
                 popicy_delay):

        self.gamma = gamma
        self.tau = tau
        self.num_actions = env.action_space.shape[0]
        self.num_states = env.observation_space.shape[0]
        self.ns = OUNoise(env.action_space)
        # TD3 updates the policy (and target networks) less frequently than the Q-function
        self.popicy_delay = popicy_delay
        self.it = 0

        if num_of_hd_layers == 1:
            self.actor = TD3Actor1(self.num_states, hidden_l1, self.num_actions)
            self.critic = TD3Critic1(self.num_actions + self.num_states, hidden_l1)
            self.actor_target = TD3Actor1(self.num_states, hidden_l1, self.num_actions)
            self.critic_target = TD3Critic1(self.num_actions + self.num_states, hidden_l1)
        elif num_of_hd_layers == 2:
            self.actor = TD3Actor2(self.num_states, hidden_l1, hidden_l2, self.num_actions)
            self.critic = TD3Critic2(self.num_actions + self.num_states, hidden_l1, hidden_l2)
            self.actor_target = TD3Actor2(self.num_states, hidden_l1, hidden_l2, self.num_actions)
            self.critic_target = TD3Critic2(self.num_actions + self.num_states, hidden_l1, hidden_l2)
        else:
            print("Wrong number of hidden layers. Use 1 or 2.")

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.memory = Memory(memory_size)
        self.loss_func = nn.MSELoss()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state, step=0, noise=False):
        """state must be a tensor
           output action must be a float number

           *** IN TRAINING MODE SET NOISE = TRUE
        """
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state).squeeze(0)
        action = action.detach().numpy()

        if noise:
            action = self.ns.get_action(action, step)
        return action

    def update(self, batch_size, step):
        # We will wait until our memory buffer has at least "batch_size" position
        if self.memory.__len__() < batch_size:
            return

        self.it += 1

        # Import samples from memory
        state, action, new_state, reward, done = self.memory.sample(batch_size)
        # If the network learned only from consecutive samples of experience as they occurred
        # sequentially in the environment, the samples would be highly correlated and would
        # therefore lead to inefficient learning. Taking random samples from replay memory breaks this correlation.

        # Convert samples to Tensors
        action = torch.FloatTensor(action)
        state = torch.FloatTensor(state)
        new_state = torch.FloatTensor(new_state)
        reward = torch.FloatTensor(reward)
        done = torch.Tensor(done)

        # Compute target_actions
        new_action = self.actor_target(new_state).detach()
        new_action = self.ns.get_action(new_action, step).float()

        Q_targ1, Q_targ2 = self.critic_target(new_state, new_action)
        target_Q = torch.min(Q_targ1, Q_targ2)
        target = reward + (1 - done) * self.gamma * target_Q.detach()

        # y(r,s',d) = r + gamma * (1-d) * min[Q_targ1(s',a'(s')), Q_targ2(s',a'(s'))]

        Q_val1, Q_val2 = self.critic(state, action)

        critic_loss1 = self.loss_func(Q_val1, target)
        critic_loss2 = self.loss_func(Q_val2, target)

        critic_loss = critic_loss1 + critic_loss2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.it % self.popicy_delay == 0:
            policy_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            # Update target network
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + (1 - self.tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
