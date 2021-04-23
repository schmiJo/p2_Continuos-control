import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor State - Action Model  Continuous input -> Continuous Output (between 1 and -1)"""

    def __init__(self, state_size=33, action_size=4, seed=546, fc1_units=64, fc2_units=64):
        """
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fn1 = nn.Linear(state_size, fc1_units)
        self.fn2 = nn.Linear(fc1_units, fc2_units)
        self.fn3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fn1(state))
        x = F.relu(self.fn2(x))
        # Use the sigmoid, because the action space is continuous from 0 to 1
        x = self.fn3(x)
        x = F.relu(x)
        return x
