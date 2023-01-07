import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, critic_learning_rate, input_size, layer1_size, layer2_size, n_actions, name, agent_dir,
                 agent_type):
        super(CriticNetwork, self).__init__()
        self.input_size = input_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.n_actions = n_actions
        self.name = name
        self.agent_dir = agent_dir
        self.agent_file = os.path.join(self.agent_dir, name + agent_type)

        self.layer1 = nn.Linear(self.input_size[0] + n_actions, self.layer1_size)
        self.layer2 = nn.Linear(self.layer1_size, self.layer2_size)
        self.output_layer = nn.Linear(self.layer2_size, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=critic_learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    # Evaluating actions
    def forward(self, state, action):
        action_value = self.layer1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.layer2(action_value)
        action_value = F.relu(action_value)

        q1 = self.output_layer(action_value)

        return q1

    def save_critic(self):
        print('saving critic')
        T.save(self.state_dict(), self.agent_file)

    def load_critic(self):
        print('loading critic')
        self.load_state_dict(T.load(self.agent_file))


class ActorNetwork(nn.Module):
    def __init__(self, actor_learning_rate, input_size, layer1_size, layer2_size, n_actions, name, agent_dir,
                 agent_type):
        super(ActorNetwork, self).__init__()
        self.input_size = input_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.n_actions = n_actions
        self.name = name
        self.agent_dir = agent_dir
        self.agent_file = os.path.join(self.agent_dir, name + agent_type)

        self.layer1 = nn.Linear(*self.input_size, self.layer1_size)
        self.layer2 = nn.Linear(self.layer1_size, self.layer2_size)
        self.output_layer = nn.Linear(self.layer2_size, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=actor_learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    # Determining what action to take
    def forward(self, state):
        prob = self.layer1(state)
        prob = F.relu(prob)
        prob = self.layer2(prob)
        prob = F.relu(prob)

        # As the action space of Lunar Lander is between +-1 no need to scale by size of max action
        prob = T.tanh(self.output_layer(prob))

        return prob

    def save_actor(self):
        print('saving actor')
        T.save(self.state_dict(), self.agent_file)

    def load_actor(self):
        print('loading actor')
        self.load_state_dict(T.load(self.agent_file))
