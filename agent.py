import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from collections import namedtuple, deque
import torch.optim as optim
from networks import ActorNet, CriticNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment"""

    def __init__(self, state_size: int, action_size: int, num_agents: int = 1, seed: int = 0):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)

    def act(self, state: np.ndarray) -> np.ndarray:
        """"Returns actions for a given state returned by the actor net
        Params
        =====
        state (array_like): current state
        """
        actions = np.random.randn(self.num_agents, self.action_size)  # select an action (for each agent)
        actions = np.clip(actions, -1, 1)
        return actions

    def step(self, state, action, reward, next_state, done) -> None:
        """Receive a state action, reward, next_state, done pair and learn from it"""

    def save_weights(self) -> None:
        """Saves all weights from all the networks the agent uses"""
        self.actor.save_weights('./weights/')
        self.critic.save_weights('./weights/')

    def restore_weights(self) -> None:
        """Restore all the saved weights from the actor and the critic"""
        self.actor.restore_weights('./weights/')
        self.critic.restore_weights('./weights/')


class Actor():

    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_sie = action_size
        self.model = ActorNet(state_size, action_size)

    def save_weights(self, path: str = './') -> None:
        """Saves the weights of the model network"""
        torch.save(self.model.state_dict(), path + 'actor')

    def restore_weights(self, path: str = './') -> None:
        """Restore the weights of the previously saved actor"""
        self.model.load_state_dict(torch.load(path + 'actor'))
        self.model.eval()

    def act(self, state: np.ndarray):
        """Act based on the state received"""

        # Create a proper pytorch tensor with the correct dimensionality
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)


class Critic():
    # used for soft update of target parameters Larger values for tau were observed to have a
    # hard time learning at the beginning but perform better eventually
    TAU: float = 0.002

    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_sie = action_size
        self.local_model = CriticNet(state_size, action_size)
        self.target_model = CriticNet(state_size, action_size)

    def save_weights(self, path: str = './') -> None:
        """Saves the weights of the local network"""
        torch.save(self.local_model.state_dict(), path + 'critic')

    def restore_weights(self, path: str = './') -> None:
        """Restore the saved local network weights to both the target and the local network"""

        self.local_model.load_state_dict(torch.load(path + 'critic'))
        self.local_model.eval()

        self.target_model.load_state_dict(torch.load(path + 'critic'))
        self.target_model.eval()
