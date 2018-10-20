## imports

# general
import numpy as np
import random
from collections import namedtuple, deque

# deep learning
import torch
import torch.nn.functional as F
import torch.optim as optim

# model architecture
from model import QNetwork

# hyperparams
from settings import BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY


# train on GPU when available
if torch.cuda.is_available():
    print("GPU available")
    device = torch.device("cuda:0")
else:
    print("No GPU available")
    device = torch.device("cpu")


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# the agent
class Agent():
    """
    Interacts with and learns from the environment
    """

    def __init__(self, state_size, action_size, seed):
        """
        Initializes agent object

        Params:

        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
        """

        # initial properties
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)  # using Adaptive Momentum optimizer

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step to 0 for updating every 'UPDATE_EVERY' steps
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        Move forward in time by one step.
        """

        # Save experience to replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn after specified time steps
        # Increment time step
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        # if it matches the update frequency
        if self.t_step == 0:
            # and if enough data is saved in memory
            if len(self.memory) > BATCH_SIZE:
                # get a random sample of experiences
                experiences = self.memory.sample()
                # update value parameters
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0):
        """
        Returns actions for a given state in accordance with the current Policy

        Params:

        state (array): Current state
        eps (float): Epsilon, for epsilon-greedy action selection
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Update value parameters using the given sampled batch of experience tuples

        Params:

        experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        gamma (float): Discount factor
        """

        # Unpack tuple of experiences
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from the target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from the local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Calculate the loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.soft_update(local_model=self.qnetwork_local, target_model=self.qnetwork_target, tau=TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update the model parameters:

        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params:
        local_model (PyTorch model): Weights are copied from
        target_model (PyTorch model): Weights are copied to
        tau (float): Interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """
    A fixed sized buffer to store experience tuples.
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initialize object

        Params:

        action_size (int): Dimension of each action
        buffer_size (int): Maximum size of buffer
        batch_size (int): Size of each training batch
        seed (int): Random seed
        """

        # initial properties
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the stored memory.
        """

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """
        Randomly sample a batch of experiences from stored memory.
        """

        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        Returns the current size of internal memory.
        """

        return len(self.memory)