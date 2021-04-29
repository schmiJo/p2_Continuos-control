import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class CriticNet(nn.Module):

    def __init__(self, state_size, action_size, fc1_size=250, fc2_size=250, fc_3_size=120, seed=0):
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
        # The second layer takes the output of the first layer as well as the action as input
        self.fc2 = nn.Linear(fc1_size + action_size, fc2_size)

        self.fc3 = nn.Linear(fc2_size, fc_3_size)
        self.fc4 = nn.Linear(fc_3_size, 1)

    def forward(self, state, action):

        print('state')
        print(state)
        print('action')
        print(action)
        x = F.relu(self.fc1(state))
        x = torch.cat((x, action.float()), dim = 1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ActorNet(nn.Module):
    def __init__(self, state_size, action_size, fc1_size=64, fc2_size=64, seed=0):
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

        self.tanh = nn.Tanh()




    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # The tanh activation function restricts the output state between -1 and 1
        return self.tanh(self.fc3(x))
