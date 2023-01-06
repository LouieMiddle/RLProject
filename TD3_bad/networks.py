import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F

# DDPG NETWORKS

"""
There are 3 Neural Networks:
    1 - with 1 Hidden Layer
    2 - with 2 Hidden Layers 

We will chose to study and compare they're results by choosing their index
"""


class DDPGCritic1(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size=1):
        super(DDPGCritic1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, output_size)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)

        return x


class DDPGActor1(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(DDPGActor1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, output_size)

    def forward(self, state):
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(x)

        return x


class DDPGCritic2(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size=1):
        super(DDPGCritic2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)

        return x


class DDPGActor2(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(DDPGActor2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)

    def forward(self, state):
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = torch.tanh(x)

        return x


# TD3_bad NETWORKS

class TD3Critic1(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size=1):
        super(TD3Critic1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, output_size)

        self.linear3 = nn.Linear(input_size, hidden_size1)
        self.linear4 = nn.Linear(hidden_size1, output_size)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = self.linear1(xu)
        x1 = F.relu(x1)
        x1 = self.linear2(x1)

        x2 = self.linear1(xu)
        x2 = F.relu(x2)
        x2 = self.linear2(x2)

        return x1, x2

    def Q1(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = self.linear1(xu)
        x1 = F.relu(x1)
        x1 = self.linear2(x1)

        return x1


class TD3Actor1(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(TD3Actor1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, output_size)

    def forward(self, state):
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(x)

        return x


class TD3Critic2(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size=1):
        super(TD3Critic2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)

        self.linear4 = nn.Linear(input_size, hidden_size1)
        self.linear5 = nn.Linear(hidden_size1, hidden_size2)
        self.linear6 = nn.Linear(hidden_size2, output_size)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = self.linear1(xu)
        x1 = F.relu(x1)
        x1 = self.linear2(x1)
        x1 = F.relu(x1)
        x1 = self.linear3(x1)

        x2 = self.linear1(xu)
        x2 = F.relu(x2)
        x2 = self.linear2(x2)
        x2 = F.relu(x2)
        x2 = self.linear3(x2)

        return x1, x2

    def Q1(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = self.linear1(xu)
        x1 = F.relu(x1)
        x1 = self.linear2(x1)
        x1 = F.relu(x1)
        x1 = self.linear3(x1)

        return x1


class TD3Actor2(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(TD3Actor2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)

    def forward(self, state):
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)

        return x
