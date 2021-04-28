import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class CriticNet(nn.Module):

    def __init__(self, state_size, action_size, fc1_size=64, fc2_size=64, seed=0):
        """Initialize the Neural Q network:
            Params:
                state_size -> size of the input layer
                action_size -> size of the output layer
                fc1 -> size of the first fully connected hidden layer
                fc2 -> size of the second fully connected hidden layer
            """
        super(CriticNet, self).__init__()

        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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

        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, action_size)

        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # The tanh activation function restricts the output state between -1 and 1
        return self.tanh(self.fc3(x))
