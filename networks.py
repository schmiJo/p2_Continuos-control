import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class CriticNet(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_size=250, fc2_size=250, fc_3_size=120):
        """Initialize the Neural Q network:
        This critic needs to map state action pairs to Q values
            Params:
                state_size -> size of the input layer
                action_size -> size of the output layer
                fc1 -> size of the first fully connected hidden layer
                fc2 -> size of the second fully connected hidden layer
            """
        super(CriticNet, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.seed = torch.manual_seed(seed)

        # The first layer only takes the states as input
        self.fc1 = nn.Linear(state_size, fc1_size)

        # batch normalization on the state
        self.bn1 = nn.BatchNorm1d(fc1_size)
        # The second layer takes the output of the first layer as well as the action as input
        self.fc2 = nn.Linear(fc1_size + action_size, fc2_size)

        self.fc3 = nn.Linear(fc2_size, fc_3_size)
        self.fc4 = nn.Linear(fc_3_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        # add an extra dim for the batch normalization
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)

        x1 = F.relu(self.fc1(state))
        x1 = self.bn1(x1)
        x = torch.cat((x1, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ActorNet(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_size=64, fc2_size=64):
        """Initialize the Actor ‚network:
            Params:
                state_size -> size of the input layer
                action_size -> size of the output layer
                fc1 -> size of the first fully connected hidden layer
                fc2 -> size of the second fully connected hidden layer
            """
        super(ActorNet, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, action_size)

        # batch normalization on the state
        self.bn1 = nn.BatchNorm1d(fc1_size)
        self.tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        # add an extra dim for the batch normalization
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)

        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        # The tanh activation function restricts the output state between -1 and 1
        return self.tanh(self.fc3(x))
