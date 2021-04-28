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

GAMMA = 0.992  # discount factor


class Agent():
    """Interacts with and learns from the environment"""

    def __init__(self, state_size: int, action_size: int, rollout_length: int = 5, num_agents: int = 1, seed: int = 0):
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
        np_actions: np.ndarray = self.actor.act(state)

        return np_actions

    def step(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done) -> None:
        """Receive a state action, reward, next_state, done pair and learn from it"""

        self.critic.step(state, reward, next_state, done)

        state_value: float = self.critic.evaluate_state(state)
        next_state_value: float = self.critic.evaluate_state(next_state)
        advantage: float = reward + GAMMA * next_state_value - state_value
        self.actor.step(state, action, reward, next_state, advantage, done)

    def save_weights(self) -> None:
        """Saves all weights from all the networks the agent uses"""
        self.actor.save_weights('./weights/')
        self.critic.save_weights('./weights/')

    def restore_weights(self) -> None:
        """Restore all the saved weights from the actor and the critic"""
        self.actor.restore_weights('./weights/')
        self.critic.restore_weights('./weights/')

    def train_critic(self, state: np.ndarray, reward: float, next_state: np.ndarray, done) -> None:
        self.critic.step(state, reward, next_state, done)

    def save_critic_weights(self) -> None:
        self.critic.save_weights('./weights/')

    def restore_critic_weights(self) -> None:
        self.critic.restore_weights('./weights/')


class Actor():
    LEARNING_RATE = 5e-4  # learning rate

    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_sie = action_size
        self.model = ActorNet(state_size, action_size)

        # The Optimizer used is Adam
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

    def save_weights(self, path: str = './') -> None:
        """Saves the weights of the model network"""
        torch.save(self.model.state_dict(), path + 'actor')

    def restore_weights(self, path: str = './') -> None:
        """Restore the weights of the previously saved actor"""
        self.model.load_state_dict(torch.load(path + 'actor'))
        self.model.eval()

    def act(self, state: np.ndarray) -> np.ndarray:
        """Act based on the state received"""

        # Create a proper pytorch tensor with the correct dimensionality
        state: torch.Tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            # Acquire an action by passing the current state to the local network
            action_values = self.model.forward(state)
        self.model.train()

        np_actions = action_values.squeeze().cpu().data.numpy()

        return np_actions

    def step(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray,
             advantage: float, done) -> None:
        """"

        action_evaluation is defined as val(next_state) - val(state)
        """
        # If the value of the state is bigger than the value of the next_state, the actor took a bad action
        # If the value of the state is smaller than the value of the next_state, the actor took a good action

        # multiplying the action_evaluation with the action vector is used to as a target to the loss function

        torch_state: torch.Tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        torch_action: torch.Tensor = self.model.forward(torch_state)
        torch_action_target = torch.from_numpy(advantage * action).float().unsqueeze(0).to(device)

        # Compute loss

        # Calculate loss
        logprob = torch.log(torch_action)
        print(logprob)
        selected_logprobs = torch.gather(logprob, 1,
                                         torch_action).squeeze()
        loss = -selected_logprobs.mean()

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Critic():
    # used for soft update of target parameters Larger values for tau were observed to have a
    # hard time learning at the beginning but perform better eventually
    TAU: float = 0.002
    BUFFER_SIZE = int(1e5)  # replay buffer size
    MINIBATCH_SIZE = 70  # minibatch size
    LEARNING_RATE = 5e-4  # learning rate
    UPDATE_EVERY = 4  # how often to update the network
    REWARD_SCALING_FACTOR = 50  # This reward scaling factor is used to overcome random noise in the network when training

    def __init__(self, state_size: int, action_size: int, seed=0):
        self.state_size = state_size
        self.action_sie = action_size

        # the critic networks should only return a value on how high the expected return is
        self.local_model = CriticNet(state_size, 1).to(device)
        self.target_model = CriticNet(state_size, 1).to(device)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.MINIBATCH_SIZE, seed)

        # The Optimizer used is Adam
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=self.LEARNING_RATE)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def save_weights(self, path: str = './') -> None:
        """Saves the weights of the local network"""
        torch.save(self.local_model.state_dict(), path + 'critic')

    def restore_weights(self, path: str = './') -> None:
        """Restore the saved local network weights to both the target and the local network"""

        self.local_model.load_state_dict(torch.load(path + 'critic'))
        self.local_model.eval()

        self.target_model.load_state_dict(torch.load(path + 'critic'))
        self.target_model.eval()

    def evaluate_state(self, state: np.ndarray) -> float:
        # Create a proper pytorch tensor with the correct dimensionality
        state: torch.Tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.local_model.eval()
        with torch.no_grad():
            # Acquire an action by passing the current state to the local network
            state_value = self.local_model.forward(state)
        self.local_model.train()
        np_state_value = state_value.squeeze().cpu().data.numpy()
        return np_state_value / self.REWARD_SCALING_FACTOR

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, rewards, next_states, dones = experiences

        # Get predicted value (for next states) from target model
        # running forward on the target network on the set of experiences
        Q_targets_next: torch.Tensor = self.target_model.forward(next_states).detach()
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected: torch.Tensor = self.local_model.forward(states)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.local_model, self.target_model, self.TAU)

    def step(self, state, reward, next_state, done):
        reward = reward * self.REWARD_SCALING_FACTOR
        # Save experience in replay memory
        self.memory.add(state, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # check if there are enough examples in memory
            if len(self.memory) > self.MINIBATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def soft_update(self, local_model, target_model, tau):
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
        self.experience = namedtuple("Experience", field_names=["state", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
