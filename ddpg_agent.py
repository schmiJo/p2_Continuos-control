import numpy as np
import random
import copy
from collections import namedtuple, deque

from networks import ActorNet, CriticNet

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
MINIBATCH_SIZE = 70  # minibatch size
GAMMA = 0.992  # discount factor
LEARNING_RATE_ACTOR = 2e-4  # learning rate for the local actor network
LEARNING_RATE_CRITIC = 3.5e-4  # learning rate for the local critic network
UPDATE_EVERY = 10  # how often to update the network
TAU = 0.002  # used for soft update of target parameters Larger values for tau were observed to have a hard time learning at the beginning but perform better eventually

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """The agent that interacts with and learns from the environment"""

    def __init__(self, state_size: int, action_size: int, seed=0):
        """Initializes an Agent object

        Params
        ======
            state_size: size of the continuos state space
            action_size: size of the continuos action space
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # --- Initialize the actor networks ---

        self.actor_local = ActorNet(state_size, action_size).to(device)
        self.actor_target = ActorNet(state_size, action_size).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=LEARNING_RATE_ACTOR)

        # --- Initialize the critic networks ---
        self.critic_local = CriticNet(state_size, action_size).to(device)
        self.critic_target = CriticNet(state_size, action_size).to(device)
        self.critic_optim = optim.Adam(self.actor_local.parameters(), lr=LEARNING_RATE_CRITIC)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, MINIBATCH_SIZE, seed)

        self.t_step = 0

    def step(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray,
             done: int) -> None:

        print('np action np np !! 👇🏼')
        print(action)
        """Save experience in replay memory"""
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # check if there are enough examples in memory
            if len(self.memory) > MINIBATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma) -> None:
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        # ---- Update the critic ----
        # Get max predicted Q values (for next states) from target model
        # running forward on the target network on the set of experiences
        next_actions = self.actor_target(next_states)

        print('next actions')
        print(next_actions)
        next_Q_targets = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (gamma * next_Q_targets * (1 - dones))


        # Get expected Q values from local model
        print('--action')
        print(actions)

        Q_expected = self.critic_local.forward(states, actions)
        # compute the loss function
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ---- update the actor ----
        action_prediction = self.actor_local.forward(states)

        print('action predictions')
        print(action_prediction)
        actor_loss: torch.Tensor = -self.critic_local(states, action_prediction).mean()
        # Minimize the loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.soft_updatee(self.critic_local, self.critic_target, TAU)
        self.soft_updatee(self.actor_local, self.actor_target, TAU)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        return np.clip(action, -1, 1)

    def hard_update(self, local_model, target_model):
        """Hard update the bias and the weights from the local Network to the target network"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def soft_updatee(self, local_model, target_model, tau):
        """Soft update the weights and biases from the local Network to the target network using the update factor tau"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        print('kukulo')
        print(actions)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
